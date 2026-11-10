# Deployment

How to run the FaultAware console (dashboard + API, login-protected) on a public server.

- **Option A — Oracle Cloud Always Free (Ampere A1, arm64) + Docker + Caddy + DuckDNS.** A permanent
  HTTPS URL for the review and after it.
- **Option B — Cloudflare quick tunnel from a laptop.** No server, no domain. Use it as the demo-day
  backup.

> **Status (2026-11-10):** The Docker stack has been tested end to end on Windows + Docker Desktop
> (fresh clone, release bundle, HTTPS login). Option A has **not** been run on Oracle Cloud yet, and the
> `linux/arm64` image build has not been verified yet (see the CI `docker` job). The steps below follow
> the tested stack. Record anything that differs in this file.

What a deployment needs:

| Piece | Where it comes from | In git? |
|---|---|---|
| Code, `Dockerfile`, `docker-compose.yml`, `deploy/Caddyfile` | this repository | yes |
| Trained models, reports, `app/demo_samples.json` | release bundle `faultaware-artifacts.zip` + `.sha256` | no |
| Login account (`secrets/auth.json`, scrypt hash only) | created on the server with `scripts/create_user.py` | no, never |

`scripts/install_artifacts.py` and `scripts/create_user.py` need only the Python standard library, so the
server's system `python3` is enough. No virtualenv is needed on the server.

---

## Option A — Oracle Cloud Always Free (arm64)

### A1. Create the VM
1. In the OCI console go to **Compute → Instances → Create instance**.
2. **Image:** Canonical Ubuntu (22.04 or 24.04). **Shape:** `VM.Standard.A1.Flex` (Ampere, arm64).
   2 OCPU / 12 GB RAM is plenty for CPU inference. Stay inside the current Always Free allowance shown in
   the console.
3. **Networking:** use the default VCN with a public subnet and **assign a public IPv4 address**.
4. **SSH keys:** upload your public key, or download the generated private key and keep it safe.
5. Boot volume: the default (≈ 47 GB) is enough.
6. After it boots, note the **public IP** and connect:
   ```bash
   ssh -i <private-key> ubuntu@<public-ip>
   ```

> If creation fails with "Out of capacity", retry later or try another availability domain. This is
> common for A1 shapes.

### A2. Open ports 80 and 443
You need to open the ports in **two** places. Both are required.

**a) VCN security list (cloud firewall).** Go to **Networking → Virtual cloud networks → your VCN →
Security Lists → Default Security List → Add Ingress Rules** and add two rules:

| Source CIDR | IP protocol | Destination port |
|---|---|---|
| `0.0.0.0/0` | TCP | `80` |
| `0.0.0.0/0` | TCP | `443` |

Port 80 is needed: Let's Encrypt validates the domain over HTTP, and Caddy redirects HTTP to HTTPS.

**b) The VM's own `iptables`.** Oracle's Ubuntu images ship a `REJECT` rule that blocks everything
except SSH. Insert the ACCEPT rules *before* that rule and save them:
```bash
sudo iptables -I INPUT 6 -m state --state NEW -p tcp --dport 80  -j ACCEPT
sudo iptables -I INPUT 6 -m state --state NEW -p tcp --dport 443 -j ACCEPT
sudo netfilter-persistent save
sudo iptables -L INPUT -n --line-numbers   # the two ACCEPT lines must appear above the REJECT line
```
If your image numbers the rules differently, choose the position so both lines land above `REJECT`.

### A3. Install Docker
```bash
sudo apt-get update && sudo apt-get upgrade -y
curl -fsSL https://get.docker.com | sudo sh        # Docker Engine + compose plugin (arm64 aware)
sudo usermod -aG docker $USER && newgrp docker     # run docker without sudo
docker run --rm hello-world && docker compose version
```

### A4. Get the code and the release bundle
```bash
git clone https://github.com/Aryan-Raj-cs27/Cloud-Fault-Prediction-BT185.git faultaware
cd faultaware
```
Download the release bundle (both files) from the GitHub Release:
```bash
curl -LO https://github.com/Aryan-Raj-cs27/Cloud-Fault-Prediction-BT185/releases/download/v1.0.0/faultaware-artifacts.zip
curl -LO https://github.com/Aryan-Raj-cs27/Cloud-Fault-Prediction-BT185/releases/download/v1.0.0/faultaware-artifacts.zip.sha256
```
If the release isn't published yet, copy the files from the training machine instead:
`scp -i <key> release/faultaware-artifacts.zip* ubuntu@<public-ip>:~/faultaware/`

Install the bundle. This verifies the SHA-256 and refuses paths outside `models/`, `reports/` and
`app/demo_samples.json`:
```bash
python3 scripts/install_artifacts.py faultaware-artifacts.zip
```

### A5. Create the admin account and fix its owner
```bash
python3 scripts/create_user.py --username admin             # prompts twice, ≥ 12 characters
# or: python3 scripts/create_user.py --username admin --generate   (prints a random password once)
sudo chown 10001:10001 secrets/auth.json
```
**The `chown` is required on Linux.** `create_user.py` writes `secrets/auth.json` with mode `0600`,
owned by your host user. The container runs as uid `10001`, so without the `chown` it can't read the
file and crash-loops with `PermissionError`. (Docker Desktop on Windows hides this, so it only shows up
on Linux. The fix has been tested.)

### A6. Point a free domain at the VM (DuckDNS)
1. Sign in at <https://www.duckdns.org>, create a subdomain (e.g. `bt185-faultaware`), and set its IP to
   the VM's **public IP**.
2. Check it from the VM: `getent hosts <name>.duckdns.org` must print the public IP.

Caddy needs a real hostname that resolves to the VM before it can get a Let's Encrypt certificate.

### A7. Start the stack
```bash
echo "DOMAIN=<name>.duckdns.org" > .env      # compose reads .env automatically (git-ignored)
docker compose up -d --build                 # first build on A1 takes several minutes
docker compose ps                            # app should become "healthy"
docker compose logs -f caddy                 # wait for "certificate obtained successfully"
```
The image is built **on the VM**, so it is a native arm64 build. No QEMU is involved.

### A8. Check it
From your laptop:
```bash
curl -I http://<name>.duckdns.org/            # 308 redirect to https
curl https://<name>.duckdns.org/api/health    # JSON: status ok, models loaded, auth configured
```
Then open `https://<name>.duckdns.org/` in a browser and sign in as `admin`.

### A9. Operate
| Task | Command (in `~/faultaware`) |
|---|---|
| Logs | `docker compose logs -f app` |
| Change password | `python3 scripts/create_user.py --username admin && sudo chown 10001:10001 secrets/auth.json && docker compose restart app` |
| Update code | `git pull && docker compose up -d --build` |
| New model bundle | download the new zip + `.sha256`, `python3 scripts/install_artifacts.py faultaware-artifacts.zip`, `docker compose restart app` |
| Stop | `docker compose down` (certificates stay in the `caddy_data` volume) |

The stack restarts by itself after a reboot (`restart: unless-stopped`, and Docker is enabled at boot).

### A10. Troubleshooting
| Symptom | Cause / fix |
|---|---|
| `app` restarting, logs show `PermissionError: .../secrets/auth.json` | Missing `sudo chown 10001:10001 secrets/auth.json` (A5). |
| Caddy logs: ACME / timeout / connection refused | Port 80 or 443 is closed in the security list or `iptables` (A2), or DuckDNS doesn't point to this IP yet (A6). |
| Browser can't connect at all, but `curl localhost` works on the VM | The `iptables` ACCEPT rules are below the `REJECT` rule (A2b). |
| Login returns 403 | The request `Origin` doesn't match the `Host`. Open the site by the same name as `DOMAIN`, not by raw IP. |
| Login returns 429 | 5 failed attempts in 15 min. Wait 15 minutes. |
| `/api/health` shows no models | Release bundle not installed (A4). Install it, then `docker compose restart app`. |

---

## Option B — Cloudflare quick tunnel (demo-day backup)

This gives a temporary public `https://<random>.trycloudflare.com` URL for a laptop, with no domain,
account or open ports. The URL changes every time you start the tunnel.

**Run the no-Docker server, not the Caddy stack.** The tunnel sends `Host: <random>.trycloudflare.com`.
Behind Caddy, the app would see `Host: localhost` while the browser sends
`Origin: https://<random>.trycloudflare.com`, so the same-origin check would reject login with 403. The
plain server gets the tunnel's `Host` directly. Tested with simulated headers: login with the tunnel
`Host` + `Origin` gives 200; a foreign `Origin` gives 403.

Prerequisites: the development setup from the README (`.venv`, release bundle installed, admin account
created) and `cloudflared` installed (<https://developers.cloudflare.com/cloudflare-one/connections/connect-networks/downloads/>).

Windows (PowerShell), terminal 1:
```powershell
$env:FAULTAWARE_COOKIE_SECURE = "1"     # the tunnel is HTTPS, so use Secure cookies
.\make.ps1 serve                        # builds the dashboard, serves dashboard + API on :8000
```
Linux/macOS: `FAULTAWARE_COOKIE_SECURE=1 make serve`.

Terminal 2:
```bash
cloudflared tunnel --url http://localhost:8000
```
Open the printed `https://….trycloudflare.com` URL and sign in. Press Ctrl+C in both terminals to stop.

Notes:
- `make serve` listens on `0.0.0.0:8000`, so devices on the same Wi-Fi can reach it too. Stop it after
  the demo.
- Quick tunnels are for demos only. They have no uptime guarantee. Keep the recorded demo video as the
  final fallback.

---

## Configuration reference
| Variable | Default | Meaning |
|---|---|---|
| `DOMAIN` | `localhost` | Hostname Caddy serves and gets a certificate for (compose only). `localhost` uses Caddy's local CA. |
| `FAULTAWARE_COOKIE_SECURE` | unset | `1` = `Secure` + `__Host-` session cookie and HSTS. Set it whenever users reach the app over HTTPS. |
| `CORS_ORIGINS` | dev origins | Extra allowed cross-origin callers. Compose sets it to `https://$DOMAIN`. |
| `FAULTAWARE_AUTH_FILE` | `secrets/auth.json` | Location of the account file. |
| `FAULTAWARE_DOCS` | unset | `1` enables the OpenAPI docs pages. Leave it off in public deployments. |
| `FAULTAWARE_ROOT`, `FAULTAWARE_FRONTEND_DIST` | set in the image | Project root and built dashboard path inside the container. |
