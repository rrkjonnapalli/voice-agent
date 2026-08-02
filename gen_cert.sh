#!/bin/bash
# Generates a self-signed TLS cert for local HTTPS testing, valid for
# localhost and this machine's current LAN IP. Re-run whenever your LAN IP
# changes or the cert expires (365 days).
set -e
LAN_IP=$(ipconfig getifaddr en0 || ipconfig getifaddr en1)
echo "Generating cert for localhost, 127.0.0.1, and $LAN_IP..."

cat > /tmp/cert_ext.cnf << EOF
[req]
distinguished_name = req_distinguished_name
x509_extensions = v3_req
prompt = no

[req_distinguished_name]
CN = localhost

[v3_req]
subjectAltName = @alt_names

[alt_names]
DNS.1 = localhost
IP.1 = 127.0.0.1
IP.2 = $LAN_IP
EOF

openssl req -x509 -newkey rsa:2048 -keyout key.pem -out cert.pem -days 365 -nodes -config /tmp/cert_ext.cnf
rm /tmp/cert_ext.cnf
echo "Done: key.pem, cert.pem"
