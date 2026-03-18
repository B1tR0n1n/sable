# SABLE Console — Electron App Setup

## Prerequisites

You need Node.js installed. If you don't have it:
- Download from https://nodejs.org/ (LTS version)
- Install with defaults

## Setup (one time)

```powershell
cd C:\Users\Tom\sable-console
npm install
```

This downloads Electron and electron-builder (~200MB).

## Run in dev mode (instant, no build)

```powershell
npm start
```

This opens the app window immediately. Use this for testing.

## Build the .exe

```powershell
npm run build
```

This creates a portable `.exe` in `dist/` — no installer needed.
Double-click to run. Self-contained, no Node.js required on the target machine.

## Connect to Nemotron

Start llama-server in another terminal:

```powershell
cd C:\Users\Tom\llama.cpp
.\build\bin\Release\llama-server.exe -m G:\models\nemotron-nano\Nemotron-3-Nano-30B-A3B-UD-Q4_K_XL.gguf --ctx-size 16384 --temp 0.6 --top-p 0.95 -ngl 999 --port 8080 --jinja
```

Then click the gear icon in the app to verify the endpoint is set to `http://127.0.0.1:8080`.

## How to Use

1. Select a topology template (Small Office, Enterprise Campus, VDI, Healthcare)
2. Click GENERATE to build the infrastructure graph
3. Click any node in the graph to inspect it
4. Click INJECT FAILURE to simulate a hardware failure on that node
5. Use the playback controls to watch the cascade propagate tick by tick
6. When the cascade finishes, click ASK NEMOTRON for AI root cause analysis
7. Click RESET to start over
