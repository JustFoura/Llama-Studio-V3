use std::net::Ipv4Addr;
use std::process::Command;

/// Find the active Tailscale IPv4 address. Do not fall back to a wildcard or
/// LAN address: when Tailscale-only mode is selected, failure must fail closed.
pub fn tailscale_ipv4() -> Result<String, String> {
    let exe = tailscale_command().ok_or_else(|| {
        "Tailscale-only mode is enabled, but the Tailscale CLI was not found. Install Tailscale, sign in, and try again.".to_string()
    })?;

    let output = Command::new(&exe)
        .args(["ip", "-4"])
        .output()
        .map_err(|e| format!("Could not run Tailscale (`{}`): {}", exe.display(), e))?;
    if !output.status.success() {
        let detail = String::from_utf8_lossy(&output.stderr).trim().to_string();
        return Err(if detail.is_empty() {
            "Tailscale did not return an address. Make sure it is running and this device is signed in.".into()
        } else {
            format!("Tailscale did not return an address: {}", detail)
        });
    }

    let stdout = String::from_utf8_lossy(&output.stdout);
    for value in stdout.split_whitespace() {
        let Ok(ip) = value.parse::<Ipv4Addr>() else {
            continue;
        };
        let octets = ip.octets();
        // Tailscale IPv4 addresses come from 100.64.0.0/10.
        if octets[0] == 100 && (64..=127).contains(&octets[1]) {
            return Ok(ip.to_string());
        }
    }

    Err("Tailscale is not connected or has no Tailscale IPv4 address. Connect this computer to your tailnet and try again.".into())
}

fn tailscale_command() -> Option<std::path::PathBuf> {
    if let Some(path) = crate::server::resolve_command("tailscale") {
        return Some(path);
    }
    #[cfg(windows)]
    {
        for root in [
            std::env::var_os("ProgramFiles"),
            std::env::var_os("ProgramW6432"),
        ]
        .into_iter()
        .flatten()
        {
            let path = std::path::PathBuf::from(root)
                .join("Tailscale")
                .join("tailscale.exe");
            if path.is_file() {
                return Some(path);
            }
        }
    }
    None
}
