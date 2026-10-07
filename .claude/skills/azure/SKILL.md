---
name: azure
description: Sign in to Azure and run Artemis's Azure operations (deploy the API, provision the database, tools or GPU VM, register resource providers). Use whenever a task needs the az CLI or touches resource group artm-rg.
---

# Azure for Artemis

## Sign in

Run `.claude/skills/azure/login.sh`. It installs the Azure CLI if missing and signs in as the `artemis-claude`
service principal from the environment variables `AZURE_CLIENT_ID`, `AZURE_CLIENT_SECRET`, `AZURE_TENANT_ID` and
(optionally) `AZURE_SUBSCRIPTION_ID`.

- Never use `az login --use-device-code`: the tenant's security defaults block it (AADSTS530035).
- If the variables are missing, tell the user to add them in the cloud environment settings and start a new
  session. Never ask for the secret in chat.

## What the service principal can do

It has Contributor on the subscription. It cannot assign roles, so the role-assignment steps in
`infra/azure/provision-db.sh` and `infra/azure/provision-tools.sh` must be run by the user in Azure Cloud Shell.
Give them the exact commands.

## Tasks

| Task | Command |
|---|---|
| Register resource providers | `bash infra/azure/register-providers.sh` |
| Deploy the website API | `infra/azure/deploy-api.sh` |
| Database (Postgres + Key Vault) | `infra/azure/provision-db.sh` (role steps: Cloud Shell) |
| Code sandbox pool | `infra/azure/provision-tools.sh` (role step: Cloud Shell) |
| GPU VM start / stop | `infra/azure/provision-gpu.sh` / `infra/azure/stop-gpu.sh` |

## Before anything that costs money

Confirm with the user first and state the price: domain purchases, `provision-gpu.sh` (about $98/hour), new
databases or VMs. After a GPU job, run `stop-gpu.sh`.
