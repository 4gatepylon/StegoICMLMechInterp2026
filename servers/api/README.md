# Stego Decoder Verification Server

TODO(hadriano) please review this.

FastAPI server that checks whether a Python source file ("cover") encodes a given secret bitstring under the project's steganographic cipher.

## API

**POST /verify**

```json
{
  "cover": "x = 1\ny = 2\n...",
  "secret": "0010110"
}
```

Returns `{"match": true}` if decoding the cover produces a bitstring that starts with `secret`. Every request sleeps 4–16 seconds before responding to throttle enumeration.

The cipher is configured server-side via the `STEGO_CIPHER` env var (default: `cipher2`).

**GET /health** — returns `{"status": "ok"}`.

## Local Setup

```bash
cd servers/api
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# default cipher is cipher2; override with:
# export STEGO_CIPHER=cipher1

uvicorn app:app --host 127.0.0.1 --port 8000
```

Test it:

```bash
curl -X POST http://127.0.0.1:8000/verify \
  -H "Content-Type: application/json" \
  -d '{"cover": "x = 1\ny = 2", "secret": "0001"}'
```

## AWS Deployment (CloudFormation)

A one-click CloudFormation template is provided in `cloudformation.yaml`.

### Prerequisites

- AWS CLI installed and configured (`aws configure`)
- An existing EC2 key pair (create one in the AWS console under EC2 > Key Pairs if needed)
- The project repo accessible from the instance (public, or use a deploy key)

### Deploy

```bash
aws cloudformation create-stack \
  --stack-name stego-verifier \
  --template-body file://cloudformation.yaml \
  --parameters \
    ParameterKey=KeyPairName,ParameterValue=YOUR_KEY_PAIR \
    ParameterKey=RepoUrl,ParameterValue=https://github.com/YOUR_ORG/StegoICMLMechInterp2026.git
```

Wait for it to finish:

```bash
aws cloudformation wait stack-create-complete --stack-name stego-verifier
```

Get the public IP and endpoint URL:

```bash
aws cloudformation describe-stacks --stack-name stego-verifier \
  --query 'Stacks[0].Outputs' --output table
```

### SSH in

```bash
ssh -i your-key.pem ec2-user@<PUBLIC_IP>
sudo journalctl -u stego-verifier -f   # view logs
```

### Tear down

```bash
aws cloudformation delete-stack --stack-name stego-verifier
```
