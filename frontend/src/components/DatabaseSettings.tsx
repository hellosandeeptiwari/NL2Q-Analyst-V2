import React, { useState, useEffect } from 'react';
import './DatabaseSettings.css';

interface DatabaseConfig {
  type: 'snowflake' | 'azure-sql' | 'postgresql';
  host: string;
  port: number;
  database: string;
  schema: string;
  username: string;
  password: string;
  // Snowflake specific
  warehouse?: string;
  account?: string;
  role?: string;
}

interface DatabaseSettingsProps {
  onNavigateBack?: () => void;
}

interface TestResult {
  success: boolean;
  message: string;
  details?: any;
}

interface IndexingStatus {
  isIndexed: boolean;
  isIndexing: boolean;
  totalTables: number;
  indexedTables: number;
  lastIndexed?: string;
  error?: string;
}

const defaultConfigs: Record<string, Partial<DatabaseConfig>> = {
  snowflake: {
    type: 'snowflake',
    schema: 'PUBLIC'
  },
  'azure-sql': {
    type: 'azure-sql',
    port: 1433,
    schema: 'dbo'
  },
  postgresql: {
    type: 'postgresql',
    port: 5432,
    schema: 'public'
  }
};

const DatabaseSettings: React.FC<DatabaseSettingsProps> = ({ onNavigateBack }) => {
  const [config, setConfig] = useState<DatabaseConfig>({
    type: 'snowflake',
    host: '',
    port: 0, // Not used for Snowflake
    database: 'COMMERCIAL_AI',
    schema: 'ENHANCED_NBA',
    username: 'SVCGCPENBADEVRW',
    password: 'Gfm2@gy@u7%HDQyoDn~N',
    warehouse: 'GCP_AI_WH',
    account: 'BTA93699-NNA65393',
    role: ''
  });

  const [testResult, setTestResult] = useState<TestResult | null>(null);
  const [isTestingConnection, setIsTestingConnection] = useState(false);
  const [isSaving, setIsSaving] = useState(false);
  const [isRefreshingStatus, setIsRefreshingStatus] = useState(false);
  const [isForceIndexing, setIsForceIndexing] = useState(false);
  const [indexingStatus, setIndexingStatus] = useState<IndexingStatus>({
    isIndexed: false,
    isIndexing: false,
    totalTables: 0,
    indexedTables: 0
  });

  useEffect(() => {
    loadSavedConfig();
    checkIndexingStatus();
  }, []);

  const loadSavedConfig = async () => {
    try {
      const response = await fetch('/api/database/config');
      if (response.ok) {
        const savedConfig = await response.json();
        if (savedConfig) {
          setConfig(savedConfig);
        }
      }
    } catch (error) {
      console.error('Failed to load saved config:', error);
    }
  };

  const checkIndexingStatus = async () => {
    setIsRefreshingStatus(true);
    try {
      console.log('Fetching indexing status...');
      const response = await fetch('/api/database/indexing-status');
      if (response.ok) {
        const status = await response.json();
        console.log('Received indexing status:', status);
        setIndexingStatus({
          isIndexed: status.isIndexed || false,
          isIndexing: status.isIndexing || false,
          totalTables: status.totalTables || 0,
          indexedTables: status.indexedTables || 0,
          lastIndexed: status.lastIndexed || null,
          error: status.error || null
        });
      } else {
        console.error('Failed to fetch indexing status:', response.status);
        setIndexingStatus(prev => ({
          ...prev,
          error: `HTTP Error: ${response.status}`
        }));
      }
    } catch (error) {
      console.error('Failed to check indexing status:', error);
      setIndexingStatus(prev => ({
        ...prev,
        error: `Network Error: ${error instanceof Error ? error.message : 'Unknown error'}`
      }));
    } finally {
      setIsRefreshingStatus(false);
    }
  };

  const forceReindex = async () => {
    setIsForceIndexing(true);
    try {
      console.log('Starting force re-index...');
      const response = await fetch('/api/database/start-indexing', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({ force_reindex: true })
      });
      if (response.ok) {
        const result = await response.json();
        console.log('Re-index triggered:', result);
        // Refresh status after triggering reindex
        setTimeout(() => {
          checkIndexingStatus();
        }, 2000); // Wait 2 seconds for the indexing to start
      } else {
        console.error('Failed to trigger re-indexing:', response.status);
      }
    } catch (error) {
      console.error('Failed to trigger re-indexing:', error);
    } finally {
      setIsForceIndexing(false);
    }
  };

  const handleTypeChange = (newType: DatabaseConfig['type']) => {
    const defaultConfig = defaultConfigs[newType];
    setConfig(prev => ({
      ...prev,
      ...defaultConfig,
      type: newType
    }));
    setTestResult(null);
  };

  const handleInputChange = (field: keyof DatabaseConfig, value: string | number) => {
    setConfig(prev => ({
      ...prev,
      [field]: value
    }));
    setTestResult(null);
  };

  const testConnection = async () => {
    setIsTestingConnection(true);
    setTestResult(null);

    try {
      const response = await fetch('/api/database/test-connection', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(config)
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const result = await response.json();
      console.log('Connection test result:', result);
      
      // Ensure the result has the proper structure
      setTestResult({
        success: result.success || false,
        message: result.message || (result.success ? 'Connection successful' : 'Connection failed'),
        details: result.connection_details || result.indexing_status || result
      });
    } catch (error) {
      console.error('Connection test error:', error);
      setTestResult({
        success: false,
        message: 'Failed to test connection',
        details: { error: error instanceof Error ? error.message : String(error) }
      });
    } finally {
      setIsTestingConnection(false);
    }
  };

  const saveConfiguration = async () => {
    setIsSaving(true);

    try {
      const response = await fetch('/api/database/config', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(config)
      });

      if (response.ok) {
        // Start auto-indexing after successful save
        await startIndexing();
        await checkIndexingStatus();
      } else {
        throw new Error('Failed to save configuration');
      }
    } catch (error) {
      console.error('Failed to save configuration:', error);
    } finally {
      setIsSaving(false);
    }
  };

  const startIndexing = async () => {
    try {
      await fetch('/api/database/start-indexing', {
        method: 'POST'
      });
    } catch (error) {
      console.error('Failed to start indexing:', error);
    }
  };

  const getDatabaseIcon = (type: string) => {
    switch (type) {
      case 'snowflake': return '❄️';
      case 'azure-sql': return '☁️';
      case 'postgresql': return '🐘';
      default: return '🗄️';
    }
  };

  const isValidForTesting = () => {
    // Common required fields
    if (!config.username || !config.database) {
      return false;
    }

    // Snowflake specific validation
    if (config.type === 'snowflake') {
      return !!(config.account && config.warehouse);
    }

    // Other databases require host
    return !!(config.host);
  };

  const renderDatabaseTypeSelector = () => (
    <div className="database-type-selector">
      <label className="database-type-label">
        Database Type
      </label>
      <div className="database-type-grid">
        {Object.keys(defaultConfigs).map((type) => (
          <button
            key={type}
            onClick={() => handleTypeChange(type as DatabaseConfig['type'])}
            className={`database-type-card ${config.type === type ? 'active' : ''}`}
          >
            <div className="database-icon">{getDatabaseIcon(type)}</div>
            <div className="database-name">
              {type.replace('-', ' ')}
            </div>
          </button>
        ))}
      </div>
    </div>
  );

  const renderFormField = (
    label: string,
    field: keyof DatabaseConfig,
    type: string = 'text',
    required: boolean = true,
    placeholder?: string
  ) => (
    <div className="form-field">
      <label className="form-label">
        {label} {required && <span className="required">*</span>}
      </label>
      <input
        type={type}
        value={config[field] || ''}
        onChange={(e) => handleInputChange(field, type === 'number' ? parseInt(e.target.value) || 0 : e.target.value)}
        placeholder={placeholder}
        className="form-input"
        required={required}
      />
    </div>
  );

  const renderConnectionForm = () => (
    <div className="connection-form">
      <div className="form-row">
        {/* Only show Host/Server for non-Snowflake databases */}
        {config.type !== 'snowflake' && renderFormField('Host/Server', 'host', 'text', true, 'server.database.windows.net')}
        {config.type !== 'snowflake' && renderFormField('Port', 'port', 'number', true)}
        
        {/* For Snowflake, show Account and Warehouse in the first row */}
        {config.type === 'snowflake' && renderFormField('Account', 'account', 'text', true, 'BTA93699-NNA65393')}
        {config.type === 'snowflake' && renderFormField('Warehouse', 'warehouse', 'text', true, 'GCP_AI_WH')}
      </div>

      <div className="form-row">
        {renderFormField('Database', 'database', 'text', true)}
        {renderFormField('Schema', 'schema', 'text', true)}
      </div>

      <div className="form-row">
        {renderFormField('Username', 'username', 'text', true)}
        {renderFormField('Password', 'password', 'password', true)}
      </div>

      {config.type === 'snowflake' && (
        <div className="form-row">
          {renderFormField('Role', 'role', 'text', false, 'Optional role')}
          {renderFormField('Port', 'port', 'number', false, '443')}
        </div>
      )}
    </div>
  );

  const renderTestResult = () => {
    if (!testResult) return null;

    return (
      <div className={`test-result ${testResult.success ? 'success' : 'error'}`}>
        <div className="test-result-header">
          <span className="text-lg">
            {testResult.success ? '✅' : '❌'}
          </span>
          <span>
            {testResult.message}
          </span>
        </div>
        {testResult.details && (
          <div className="test-result-details">
            <pre>{JSON.stringify(testResult.details, null, 2)}</pre>
          </div>
        )}
      </div>
    );
  };

  const renderIndexingStatus = () => (
    <div className="indexing-section">
      <h3 className="section-header">
        <span>🔍</span>
        <span>Vector Search Indexing</span>
      </h3>

      <div className="indexing-content">
        <div className="status-row">
          <span className="status-label">Status:</span>
          <span className={`status-badge ${
            indexingStatus.isIndexing 
              ? 'indexing'
              : indexingStatus.isIndexed
              ? 'indexed'
              : 'not-indexed'
          }`}>
            {indexingStatus.isIndexing 
              ? 'Indexing...' 
              : indexingStatus.isIndexed 
              ? 'Indexed' 
              : 'Not Indexed'}
          </span>
        </div>

        {indexingStatus.totalTables > 0 && (
          <div className="status-row">
            <span className="status-label">Tables:</span>
            <span>
              {indexingStatus.indexedTables} / {indexingStatus.totalTables}
            </span>
          </div>
        )}

        {indexingStatus.lastIndexed && (
          <div className="status-row">
            <span className="status-label">Last Indexed:</span>
            <span>{new Date(indexingStatus.lastIndexed).toLocaleString()}</span>
          </div>
        )}

        {indexingStatus.error && (
          <div className="error-message">
            {indexingStatus.error}
          </div>
        )}

        <div className="indexing-actions">
          <button
            onClick={checkIndexingStatus}
            disabled={indexingStatus.isIndexing || isRefreshingStatus}
            className="refresh-button"
          >
            {isRefreshingStatus ? 'Refreshing...' : indexingStatus.isIndexing ? 'Indexing...' : 'Refresh Status'}
          </button>
          
          <button
            onClick={forceReindex}
            disabled={indexingStatus.isIndexing || isRefreshingStatus || isForceIndexing}
            className="force-reindex-button"
          >
            {isForceIndexing ? 'Starting Re-index...' : 'Force Re-index All Tables'}
          </button>
        </div>
      </div>
    </div>
  );

  return (
    <div className="database-settings">
      <div className="database-settings-container">
        {/* Header */}
        <div className="database-settings-header">
          <div className="header-title">
            <span className="header-icon">⚙️</span>
            <h1>Database Settings</h1>
          </div>
          {onNavigateBack && (
            <button
              onClick={onNavigateBack}
              className="back-button"
            >
              <span>←</span>
              <span>Back to Chat</span>
            </button>
          )}
        </div>

        <div className="settings-content">
          {/* Database Connection Section */}
          <div className="connection-section">
            <h2 className="section-header">
              <span>🗄️</span>
              <span>Database Connection</span>
            </h2>

            {renderDatabaseTypeSelector()}
            {renderConnectionForm()}

            {/* Test Result */}
            {testResult && renderTestResult()}

            {/* Action Buttons */}
            <div className="action-buttons">
              <button
                onClick={testConnection}
                disabled={isTestingConnection || !isValidForTesting()}
                className="action-button test-button"
              >
                {isTestingConnection && (
                  <div className="loading-spinner"></div>
                )}
                <span>{isTestingConnection ? 'Testing...' : 'Test Connection'}</span>
              </button>

              <button
                onClick={saveConfiguration}
                disabled={isSaving || !testResult?.success}
                className="action-button save-button"
              >
                {isSaving && (
                  <div className="loading-spinner"></div>
                )}
                <span>{isSaving ? 'Saving...' : 'Save Configuration'}</span>
              </button>
            </div>

            {!testResult?.success && testResult && (
              <p className="help-text">
                <span>💡</span>
                <span>Please test the connection successfully before saving the configuration.</span>
              </p>
            )}
          </div>

          {/* Vector Search Indexing Section */}
          {renderIndexingStatus()}

          {/* Usage Instructions */}
          <div className="instructions-section">
            <h3 className="instructions-header">
              <span>💡</span>
              <span>How it works</span>
            </h3>
            <div className="instructions-list">
              <p>1. Select your database type and enter connection details</p>
              <p>2. Test the connection to ensure it's working properly</p>
              <p>3. Save the configuration to enable auto-indexing</p>
              <p>4. The system will automatically create vector embeddings for schema discovery</p>
              <p>5. Use natural language queries in the chat to find and analyze data</p>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default DatabaseSettings;
