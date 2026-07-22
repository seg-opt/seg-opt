# Connecting to PCSS Eagle

```bash
ssh -i <private_key> <username>@eagle.man.poznan.pl
```

## What the command contains

| Part                  | Meaning                                                                              |
| --------------------- | ------------------------------------------------------------------------------------ |
| `ssh`                 | OpenSSH client — opens a secure shell to the remote host.                            |
| `-i <private_key>`    | Path to your **private SSH key** (its public half is registered on the cluster).     |
| `<username>`          | Your cluster login (e.g. `jan-kowalski`) — set as `USERNAME` in `.env.test`.         |
| `eagle.man.poznan.pl` | The Eagle **login node** address — set as `CLUSTER_ADDRESS` in `.env.test`.          |

## Using the helper script

Instead of running `ssh` by hand, use [`scripts/connect.sh`](../scripts/connect.sh):

```bash
bash scripts/connect.sh
```

It reads `SSH_KEY`, `USERNAME`, and `CLUSTER_ADDRESS` from `.env.test` (see
[`.env.example`](../.env.example) for the format), writes the key to a
temporary `0600` file, opens the SSH session, and removes the temp key on exit.
