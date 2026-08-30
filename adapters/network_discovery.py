#!/usr/bin/env python3
"""
SABLE network auto-discovery (nmap + SNMP/LLDP).

Two-tier scan that emits a SABLE topology JSON the model can diagnose:
  1. nmap  — host sweep + service/version fingerprinting -> sharp role inference
  2. SNMP/LLDP/CDP — for managed gear, pull real neighbour tables -> true L2 edges
Falls back gracefully: no SNMP -> L3 star (hosts -> gateway/DNS); no nmap -> TCP probe.

Honest note: nmap OS detection (-O) and SYN scan need root; run with sudo for
richer fingerprints. MAC/vendor comes from the ARP table. Inferred roles are
guesses (service+MAC+hostname), each recorded in `inferred_by`.

    python3 network_discovery.py                 # scan local /24, print + save topology
    python3 network_discovery.py --diagnose      # scan + run SABLE on it
    python3 network_discovery.py --snmp public    # try SNMP community 'public' for L2
    sudo -E python3 network_discovery.py          # richer (OS detect + MAC)
"""
import argparse, ipaddress, json, re, socket, subprocess, sys
from pathlib import Path

try:
    import nmap
    HAVE_NMAP = True
except Exception:
    HAVE_NMAP = False

OUI = {  # MAC prefix -> role hint (net/fw/hyp/srv/stor)
    "00:00:0C":"net","00:1A:A1":"net","00:0B:86":"net","6C:F3:7F":"net","00:05:85":"net",
    "3C:8A:B0":"net","24:5A:4C":"net","74:AC:B9":"net","FC:EC:DA":"net","78:8A:20":"net",
    "68:D7:9A":"net","04:18:D6":"net","4C:5E:0C":"net","48:8F:5A":"net","CC:2D:E0":"net",
    "DC:2C:6E":"net","00:09:5B":"net","A0:40:A0":"net","50:C7:BF":"net","EC:08:6B":"net",
    "00:09:0F":"fw","90:6C:AC":"fw","00:1B:17":"fw",
    "00:50:56":"hyp","00:0C:29":"hyp","00:05:69":"hyp",
    "00:14:22":"srv","B8:2A:72":"srv","F8:BC:12":"srv","00:1B:78":"srv","3C:D9:2B":"srv",
    "98:4B:E1":"srv","00:25:90":"srv","0C:C4:7A":"srv","AC:1F:6B":"srv","B8:27:EB":"srv",
    "DC:A6:32":"srv","E4:5F:01":"srv",
    "00:11:32":"stor","90:09:D0":"stor","00:08:9B":"stor","24:5E:BE":"stor","00:A0:98":"stor",
}
ROLE_TYPE = {"net":"ACCESS_SWITCH","fw":"FIREWALL","hyp":"HYPERVISOR","srv":"SERVER_PHYSICAL",
             "stor":"STORAGE_ARRAY","dns":"DNS_SERVER","dc":"DOMAIN_CONTROLLER",
             "app":"APPLICATION_SERVICE","gw":"INTERNET_GATEWAY","router":"CORE_SWITCH"}


def sh(cmd, t=10):
    try: return subprocess.run(cmd, capture_output=True, text=True, timeout=t).stdout
    except Exception: return ""


def safe_id(s, fallback="node"):
    """Reduce a network-derived string (hostname, SNMP name) to a safe node id.

    Scanned hostnames are attacker-controllable (DHCP option 12, NetBIOS, mDNS).
    They flow into node ids that the dashboard renders, so strip to a strict
    charset here — no HTML metacharacters can survive into the topology JSON.
    """
    clean = re.sub(r"[^A-Za-z0-9_-]", "", (s or "").strip())[:40]
    return clean or fallback


def local_config():
    gw = src = subnet = None; dns = []
    m = re.search(r"default via (\S+).*?src (\S+)", sh(["ip", "route"]))
    if m: gw, src = m.group(1), m.group(2)
    if src:
        pfx = ".".join(src.split(".")[:3])
        for line in sh(["ip", "-brief", "addr"]).splitlines():
            mm = re.search(rf"({re.escape(pfx)}\.\d+/\d+)", line)
            if mm: subnet = mm.group(1)
    for line in sh(["resolvectl", "status"]).splitlines():
        if "DNS Server" in line:
            dns += [ip for ip in re.findall(r"\d+\.\d+\.\d+\.\d+", line) if not ip.startswith("127.")]
    return {"gateway": gw, "self": src, "subnet": subnet, "dns": sorted(set(dns))}


def arp_map():
    m = {}
    for line in sh(["ip", "neigh"]).splitlines():
        p = line.split()
        if len(p) >= 5 and p[0].count(".") == 3 and "lladdr" in p:
            m[p[0]] = p[p.index("lladdr") + 1].upper()
    return m


def nmap_scan(subnet):
    """nmap sweep + service scan. Returns {ip: {hostname, ports:[(p,name,product)], osmatch, devtype}}."""
    nm = nmap.PortScanner()
    print(f"  nmap -sn sweep {subnet} ...", file=sys.stderr)
    nm.scan(hosts=subnet, arguments="-sn -T4")
    live = [h for h in nm.all_hosts() if nm[h].state() == "up"]
    print(f"  {len(live)} up; nmap -sV service scan ...", file=sys.stderr)
    root = (subprocess.run(["id","-u"],capture_output=True,text=True).stdout.strip() == "0")
    args = ("-sV -O -T4 --top-ports 100 --host-timeout 40s" if root
            else "-sT -sV -T4 --top-ports 100 --host-timeout 40s")
    out = {}
    if live:
        nm.scan(hosts=" ".join(live), arguments=args)
        for h in live:
            if h not in nm.all_hosts(): out[h] = {"hostname":"","ports":[],"osmatch":"","devtype":""}; continue
            host = nm[h]
            ports = []
            for proto in host.all_protocols():
                for p in host[proto]:
                    d = host[proto][p]
                    if d.get("state") == "open":
                        ports.append((p, d.get("name",""), (d.get("product","")+" "+d.get("version","")).strip()))
            osm = host["osmatch"][0]["name"] if host.get("osmatch") else ""
            dt = ""
            if host.get("osmatch") and host["osmatch"][0].get("osclass"):
                dt = host["osmatch"][0]["osclass"][0].get("type","")
            out[h] = {"hostname": host.hostname(), "ports": ports, "osmatch": osm, "devtype": dt}
    return out


def snmp_neighbors(ip, community):
    """Try SNMP: sysDescr + LLDP/CDP neighbour system names. Returns (sysdescr, [neighbor_names])."""
    def walk(oid):
        return sh(["snmpwalk","-v2c","-c",community,"-t","1","-r","0","-On",ip,oid], t=6)
    sysd = sh(["snmpget","-v2c","-c",community,"-t","1","-r","0","-Ovq",ip,"1.3.6.1.2.1.1.1.0"], t=4).strip()
    if not sysd:
        return None, []
    names = []
    for oid in ("1.0.8802.1.1.2.1.4.1.1.9",           # LLDP lldpRemSysName
                "1.3.6.1.4.1.9.9.23.1.2.1.1.6"):       # CDP cdpCacheDeviceId
        for line in walk(oid).splitlines():
            m = re.search(r'"?([^"\s][^"]*?)"?\s*$', line.split("=")[-1].strip())
            if m and m.group(1) and "No Such" not in line:
                names.append(m.group(1).split(".")[0])
    return sysd, sorted(set(n for n in names if n))


def classify(ip, mac, host_info, sysd, is_gw, is_dns):
    if is_gw: return "INTERNET_GATEWAY", "default route"
    ports = host_info.get("ports", []); names = " ".join(n for _,n,_ in ports)
    prods = " ".join(p for _,_,p in ports).lower()
    portnums = {p for p,_,_ in ports}
    h = (host_info.get("hostname","") or "").lower()
    dt = (host_info.get("devtype","") or "").lower()
    blob = (sysd or "").lower() + " " + prods
    # strongest signals first
    if "esxi" in blob or "vmware" in prods or 902 in portnums: return "HYPERVISOR", "VMware/ESXi"
    if "synology" in blob or "qnap" in blob or 5000 in portnums or "iscsi" in names or "nfs" in names: return "STORAGE_ARRAY", "storage service"
    if {88,389,636}.intersection(portnums) or ("kerberos" in names and "ldap" in names): return "DOMAIN_CONTROLLER", "AD/LDAP/Kerberos"
    if 53 in portnums or "domain" in names or is_dns: return "DNS_SERVER", "DNS (53)"
    if dt in ("switch","router","wap","bridge") or "switch" in blob or "router" in blob: return ("CORE_SWITCH" if "router" in (dt+blob) else "ACCESS_SWITCH"), "nmap/SNMP device type"
    if "fortigate" in blob or "palo alto" in blob or "firewall" in blob: return "FIREWALL", "firewall banner"
    for kw,t in [("dc","DOMAIN_CONTROLLER"),("dns","DNS_SERVER"),("esx","HYPERVISOR"),
                 ("nas","STORAGE_ARRAY"),("san","STORAGE_ARRAY"),("sw","ACCESS_SWITCH"),
                 ("fw","FIREWALL"),("host","HYPERVISOR")]:
        if kw in h: return t, f"hostname '{h}'"
    if {80,443,8080}.intersection(portnums): return "APPLICATION_SERVICE", "web service"
    v = OUI.get(mac[:8]) if mac else None
    if v: return ROLE_TYPE.get(v,"SERVER_PHYSICAL"), "MAC vendor"
    if portnums: return "SERVER_PHYSICAL", f"{len(portnums)} open ports"
    return "SERVER_PHYSICAL", "host up"


def discover(community=None):
    cfg = local_config()
    print(f"  self={cfg['self']} gw={cfg['gateway']} subnet={cfg['subnet']} dns={cfg['dns']}", file=sys.stderr)
    if not HAVE_NMAP:
        print("  [!] python-nmap/nmap missing — install for full discovery", file=sys.stderr); sys.exit(1)
    scan = nmap_scan(cfg["subnet"])
    arp = arp_map()
    dns_set = set(cfg["dns"])

    # SNMP pass (managed gear -> real neighbours)
    snmp = {}
    if community:
        print(f"  SNMP pass (community '{community}') ...", file=sys.stderr)
        for ip, info in scan.items():
            if 161 in {p for p,_,_ in info["ports"]} or True:  # try all; cheap timeout
                sysd, nbrs = snmp_neighbors(ip, community)
                if sysd: snmp[ip] = {"sysdescr": sysd, "neighbors": nbrs}
        print(f"  {len(snmp)} host(s) answered SNMP", file=sys.stderr)

    ids, comps, obs = {}, [], []
    for ip in sorted(scan, key=lambda x: ipaddress.ip_address(x)):
        info = scan[ip]; mac = arp.get(ip, ""); sysd = snmp.get(ip,{}).get("sysdescr")
        ctype, why = classify(ip, mac, info, sysd, ip == cfg["gateway"], ip in dns_set)
        base = safe_id(info["hostname"].split(".")[0] if info["hostname"]
                       else ctype.split("_")[0].lower(), fallback="node")
        nid = f"{base}-{ip.split('.')[-1]}"
        ids[ip] = nid
        comps.append({"id": nid, "type": ctype, "ip": ip, "mac": mac,
                      "hostname": info["hostname"], "os": info.get("osmatch",""),
                      "ports": [p for p,_,_ in info["ports"]], "inferred_by": why})
        obs.append({"id": nid, "state": "healthy"})

    gw_id = ids.get(cfg["gateway"])
    if not gw_id and cfg["gateway"]:
        gw_id = "gateway-" + cfg["gateway"].split(".")[-1]
        comps.append({"id": gw_id, "type": "INTERNET_GATEWAY", "ip": cfg["gateway"], "inferred_by": "default route"})
        obs.append({"id": gw_id, "state": "healthy"}); ids[cfg["gateway"]] = gw_id
    comps.append({"id": "wan-1", "type": "WAN_LINK", "inferred_by": "gateway uplink"})
    obs.append({"id": "wan-1", "state": "healthy"})

    deps = []
    if gw_id: deps.append({"source": gw_id, "target": "wan-1", "type": "NETWORK_PATH"})
    dns_ids = [ids[d] for d in cfg["dns"] if d in ids]
    # L2 edges from SNMP/LLDP where we have them
    name_to_id = {c["hostname"].split(".")[0].lower(): c["id"] for c in comps if c.get("hostname")}
    l2 = set()
    for ip, s in snmp.items():
        for nb in s["neighbors"]:
            tid = name_to_id.get(nb.lower())
            if tid and tid != ids[ip]: l2.add((ids[ip], tid))
    for a, b in l2: deps.append({"source": a, "target": b, "type": "NETWORK_PATH", "via": "LLDP/CDP"})
    l2_nodes = {x for e in l2 for x in e}
    # L3 fallback edges for nodes without an L2 uplink
    for ip, nid in ids.items():
        if nid == gw_id or nid == "wan-1": continue
        if nid not in l2_nodes and gw_id:
            deps.append({"source": nid, "target": gw_id, "type": "NETWORK_PATH"})
        for did in dns_ids:
            if did != nid: deps.append({"source": nid, "target": did, "type": "DNS_DEPENDENCY"})

    return {"name": "auto-discovered-" + (cfg["subnet"] or "net"), "components": comps,
            "dependencies": deps, "observations": obs,
            "meta": {"gateway": cfg["gateway"], "subnet": cfg["subnet"], "dns": cfg["dns"],
                     "snmp_hosts": list(snmp), "l2_edges": len(l2)}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out"); ap.add_argument("--snmp", metavar="COMMUNITY", default=None)
    ap.add_argument("--diagnose", action="store_true")
    args = ap.parse_args()

    topo = discover(community=args.snmp)
    print(f"\n  Discovered {len(topo['components'])} components, {len(topo['dependencies'])} deps"
          f" ({topo['meta']['l2_edges']} real L2 edges via LLDP/CDP)\n")
    print(f"    {'id':24} {'type':20} {'ip':16} inferred by")
    print("    " + "-" * 82)
    for c in topo["components"]:
        print(f"    {c['id']:24} {c['type']:20} {c.get('ip',''):16} {c.get('inferred_by','')}")

    out = Path(args.out) if args.out else Path(__file__).parent / "discovered_topology.json"
    clean = {"name": topo["name"],
             "components": [{"id": c["id"], "type": c["type"]} for c in topo["components"]],
             "dependencies": [{k: e[k] for k in ("source","target","type") if k in e} for e in topo["dependencies"]],
             "observations": topo["observations"]}
    out.write_text(json.dumps(clean, indent=2))
    (out.with_name("discovered_full.json")).write_text(json.dumps(topo, indent=2))
    print(f"\n  [✓] SABLE topology: {out}")

    if args.diagnose:
        sys.path.insert(0, str(Path(__file__).parent))
        import topology_ingest as TI
        print(); TI.diagnose(str(out))


if __name__ == "__main__":
    main()
