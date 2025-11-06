# Azure App Service Environment Variables Checklist

## Database Configuration (Azure SQL)
- ✅ AZURE_SQL_HOST = odsproduction.database.windows.net
- ✅ AZURE_SQL_PORT = 1433
- ✅ AZURE_SQL_USER = odsjobsuser
- ✅ AZURE_SQL_PASSWORD = DwHIBSAOD$J0bs!1
- ✅ AZURE_SQL_DATABASE = DWHPRODIBSA
- ✅ AZURE_SQL_DRIVER = ODBC Driver 17 for SQL Server
- ✅ DB_ENGINE = azure_sql
- ⚠️ AZURE_SCHEMA = (need to add if missing)

## OpenAI Configuration
- ✅ OPENAI_API_KEY = (configured)
- ✅ OPENAI_MODEL = gpt-4o-mini
- ⚠️ REASONING_MODEL = (check if added)
- ⚠️ USE_REASONING_FOR_PLANNING = (check if added)
- ⚠️ EMBED_MODEL = (check if added)

## Pinecone Configuration
- ✅ PINECONE_API_KEY = pcsk_3nnsmY_7bkkfaxVBt21Q9Xz9tVX1XzMZE84uqrpxh2TkmpXDUYrSe2cCVwVhXwvdrDbqCC
- ✅ PINECONE_ENVIRONMENT = us-west1-gcp
- ✅ PINECONE_INDEX_NAME = nl2q-schema-stiwar12

## Azure Search Configuration
- ✅ AZURE_SEARCH_ENDPOINT = https://aianalystagentcai.search.windows.net
- ✅ AZURE_SEARCH_KEY = (configured)
- ✅ AZURE_SEARCH_INDEX_NAME = nl2q-schema-index

## Additional Settings
- ⚠️ SKIP_VECTOR_SEARCH = true (check if added)
- ⚠️ STORAGE_TYPE = local (check if added)
- ⚠️ FAST_STARTUP = true (check if added)

## Indexing Configuration
- ✅ INDEX_TABLE_BATCH_SIZE
- ✅ INDEX_EMBEDDING_BATCH_SIZE
- ✅ INDEX_UPSERT_BATCH_SIZE
- ✅ INDEX_SKIP_ROW_COUNTS

## Build Configuration
- ✅ SCM_DO_BUILD_DURING_DEPLOYMENT = true
- ✅ ENABLE_ORYX_BUILD = true

---

## How to Verify/Add Missing Variables:

1. Go to Azure Portal: https://portal.azure.com
2. Navigate to: l2q-analyst-backend → Settings → Environment variables
3. Check if ALL variables above are present
4. Add any missing ones (marked with ⚠️)
5. Click "Save" and "Restart" the app

## Testing After Configuration:

Test these endpoints to verify:
1. https://l2q-analyst-backend-ayffadegfschjcs.eastus-01.azurewebsites.net/health
2. https://l2q-analyst-backend-ayffadegfschjcs.eastus-01.azurewebsites.net/api/database/status
3. https://l2q-analyst-backend-ayffadegfschjcs.eastus-01.azurewebsites.net/docs
