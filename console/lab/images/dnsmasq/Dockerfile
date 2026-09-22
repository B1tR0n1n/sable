# The lab's DNS server: dnsmasq on Alpine, wrapped by an entrypoint that keeps the
# "cache" (a runtime hosts file) separate from the pristine records so that
# `kill -s HUP` (catalog: clear_dns_cache) and a restart (catalog: restart_service)
# both clear a poisoned record. See entrypoint.sh and ../../faults/poison_dns.sh.
FROM alpine:3.20
RUN apk add --no-cache dnsmasq
COPY entrypoint.sh /entrypoint.sh
RUN chmod 0755 /entrypoint.sh && mkdir -p /run/dnsmasq
EXPOSE 53/udp 53/tcp
ENTRYPOINT ["/entrypoint.sh"]
