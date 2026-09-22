# OVERLORD × SABLE Console — UI (Phase 7)

The single operator screen. React 18 + TypeScript + Vite; no UI kit, no state
library. It talks only to the console server (`console/server/`, FastAPI,
`127.0.0.1:7780`) using the contract in `console/docs/API.md` and mirrors the
Phase 1 schemas (`console/contracts/schema/*.json`) in `src/types.ts`.

## Dev

```sh
cd console/ui
npm install
npm run dev          # http://localhost:5173 — proxies /api and /ws to 127.0.0.1:7780
```

Start the console server first (it must be listening on 7780); the UI polls
`/api/state` every 5 s while the `/ws` socket is down and reconnects with
exponential backoff.

## Build

```sh
npm run build        # type-checks, then writes dist/
```

The FastAPI server serves `console/ui/dist/` at `/`. The receipt view is a
client-side route (`/receipts/:id`); the server must fall back to
`index.html` for unknown paths, or open it as `/#/receipts/:id`.

## Test

```sh
npm test             # Vitest + @testing-library/react (jsdom); fetch/WebSocket are mocked
```

Fixtures under `src/__tests__/fixtures/` are copies of
`console/contracts/fixtures/*.json`.

## Layout

| Region | Component | Data |
|---|---|---|
| Loop-state strip | `LoopStrip` | `GET /api/state`, `hello`/`state` events; stations lit from `finding`/`plan`/`countdown`/`step`/`verification`/`receipt` events |
| Findings | `FindingsList` | `GET /api/findings?status=open\|closed`, `finding` events |
| Finding · Plan · Gate | `PlanPanel` | `GET /api/findings/{id}/plan`, `plan`/`step`/`approval`/`countdown` events; `POST /api/plans/{id}/approve`, `POST /api/findings/{id}/plan {planner:"llm"}` |
| Autonomy matrix | `AutonomyMatrix` | `GET /api/policy`, `PUT /api/policy` (target matrix, confirmed) |
| Session log | `SessionLog` | `GET /api/log?limit=200`, `log` events |
| Receipts | `ReceiptsList`, `ReceiptView` | `GET /api/receipts`, `receipt` events; `/api/receipts/{id}`, `/export?format=json\|md`, `/api/receipts/verify` |
| Analyst drawer | `AnalystPanel` | `POST /api/analyst/chat` |
| Lab | `LabControls` (only when `state.lab`) | `POST /api/lab/fault {name}` |

The operator name used as `actor` on approvals is kept in
`localStorage["console.operator"]`.

## Identity

b1tr0n1n / Field Systems Division: dark ground only (`#0a0908` / `#0f0e0b`),
warm text (`#c8bda0` / `#ede5d0`), gold `#c9a227` as the only accent, red
`#a63d2f` and green `#4a7a45` for semantics only. Cormorant Garamond for
headings and prose, JetBrains Mono for labels, metadata and logs; 80 px grid
under everything. Tokens live in `src/styles.css`. Fonts are loaded via a
`<link>` in `index.html` with a local fallback stack (Georgia / Consolas).
