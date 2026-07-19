# Connecting to PCSS Eagle

```bash
ssh -i <private_key> <username>@eagle.man.poznan.pl
```

Or just run the helper, which reads everything from `.env`:

```bash
./connect.sh
```

## What the command contains

| Part                   | Meaning                                                        |
| ---------------------- | ------------------------------------------------------------- |
| `ssh`                  | OpenSSH client — opens a secure shell to the remote host.     |
| `-i <private_key>`     | Path to your **private SSH key** (its public half is registered on the cluster). |
| `<username>`           | Your cluster login (e.g. `jan-kowalski`) — set as `USERNAME` in `.env`. |
| `eagle.man.poznan.pl`  | The Eagle **login node** address — set as `CLUSTER_ADDRESS` in `.env`. |

`connect.sh` writes the `SSH_KEY` from `.env` to a temporary file (permissions
`600`), runs the `ssh` command above, then deletes the temp key on exit.
