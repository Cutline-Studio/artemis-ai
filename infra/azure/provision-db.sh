#!/bin/bash
# Database for the website API: PostgreSQL Flexible Server + Key Vault holding its connection URL.
# The API reads ARTEMIS_DATABASE_URL from the Key Vault secret through the container app's managed identity.
# East US blocks new PostgreSQL servers on this subscription, so the database lives in East US 2.
set -euo pipefail
RG=${RG:-artm-rg}
DB=${DB:-artemis-db-6b22c1}
KV=${KV:-artemis-kv-6b22c1}
APP=${APP:-artemis-api}
PW=$(python3 -c "import secrets,string;a=string.ascii_letters+string.digits;print('Ar'+''.join(secrets.choice(a) for _ in range(30))+'9x')")

az postgres flexible-server create -g "$RG" -n "$DB" -l eastus2 --tier Burstable --sku-name Standard_B1ms \
  --version 16 --storage-size 32 --admin-user artemisadmin --admin-password "$PW" \
  --public-access 0.0.0.0 --high-availability Disabled --yes
az postgres flexible-server db create -g "$RG" -s "$DB" -d artemis

az keyvault create -g "$RG" -n "$KV" -l eastus --enable-rbac-authorization true --retention-days 7
az role assignment create --assignee "$(az account show --query user.name -o tsv)" --role "Key Vault Secrets Officer" \
  --scope "$(az keyvault show -n "$KV" --query id -o tsv)"
URL="postgresql://artemisadmin:$PW@$DB.postgres.database.azure.com:5432/artemis?sslmode=require"
az keyvault secret set --vault-name "$KV" -n artemis-database-url --value "$URL" -o none

az containerapp identity assign -g "$RG" -n "$APP" --system-assigned
az role assignment create --assignee "$(az containerapp show -g "$RG" -n "$APP" --query identity.principalId -o tsv)" \
  --role "Key Vault Secrets User" --scope "$(az keyvault show -n "$KV" --query id -o tsv)/secrets/artemis-database-url"
az containerapp secret set -g "$RG" -n "$APP" \
  --secrets "database-url=keyvaultref:https://$KV.vault.azure.net/secrets/artemis-database-url,identityref:system"
az containerapp update -g "$RG" -n "$APP" \
  --set-env-vars ARTEMIS_DATABASE_URL=secretref:database-url ARTEMIS_TRUST_PROXY=1 --remove-env-vars ARTEMIS_DB
