# Supermarket System Security & Compliance Policy (OWASP ASVS & ATT&CK)

## 1. Secure Data Handling & Sandboxing
- All uploaded inventory and sales CSV files must be processed through strict schema validation and column types parsing to prevent injection attacks.
- File parser libraries (e.g., pandas, csv) are executed within sandbox constraints with restricted read/write access to system environment variables.

## 2. OWASP ASVS Compliance
- **V3: Session Management**: No state is persisted on server environments between sessions. User state is isolated to Streamlit local web session memory.
- **V5: Validation, Sanitization and Encoding**: All chatbot text inputs and search queries undergo standard string sanitization before being embedded or matched.
- **V12: File Upload Security**: Uploaded CSV files are checked for size limits (< 5MB) and strictly verified for CSV mime-type and structure columns.

## 3. Threat Modeling (CVSS & ATT&CK Mitigation)
- **Data Poisoning Mitigation**: ChromaDB vector databases are isolated behind local credentials. Only verified knowledge base documents from trusted administrator folders are indexed.
- **Inference Attack Protection**: The local custom LLM runs entirely offline. Prompts, context queries, and RAG metadata are never transmitted over public networks, mitigating data leakage.
