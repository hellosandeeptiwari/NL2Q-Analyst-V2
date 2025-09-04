import React, { useState, useEffect } from 'react';
import axios from 'axios';

interface DatabaseConfig {
  type: 'snowflake' | 'postgres' | 'mysql' | 'azure_sql';
  host: string;
  port: string;
  database: string;
  username: string;
  password: string;
  // Snowflake specific
  account?: string;
  warehouse?: string;
  schema?: string;
  role?: string;
  // Azure SQL specific
  driver?: string;
}

interface SettingsPanelProps {
  isOpen: boolean;
  onClose: () => void;
  onDatabaseChange: (dbType: string) => void;
}

function SettingsPanel({ isOpen, onClose, onDatabaseChange }: SettingsPanelProps) {
  const [activeTab, setActiveTab] = useState<'database' | 'general'>('database');
  const [dbConfig, setDbConfig] = useState<DatabaseConfig>({
    type: 'snowflake',
    host: '',
    port: '443',
    database: '',
    username: '',
    password: ''
  });
  const [isConnecting, setIsConnecting] = useState(false);
  const [connectionStatus, setConnectionStatus] = useState<'idle' | 'success' | 'error'>('idle');
  const [statusMessage, setStatusMessage] = useState('');

  useEffect(() => {
    const loadConfiguration = async () => {
      console.log('🔧 Loading database configuration...');
      
      try {
        // Always try to fetch from backend environment variables first
        console.log('📡 Fetching configuration from backend...');
        const response = await axios.get('http://localhost:8003/get-db-config');
        console.log('✅ Backend response:', response.data);
        
        if (response.data && response.data.config) {
          const envConfig = response.data.config;
          
          // Pre-populate with Snowflake config from environment if available
          if (envConfig.snowflake && 
              (envConfig.snowflake.account || envConfig.snowflake.username || envConfig.snowflake.database)) {
            console.log('🔗 Found Snowflake config in environment, populating form...');
            setDbConfig(prev => ({
              ...prev,
              type: 'snowflake',
              host: envConfig.snowflake.account || '',
              database: envConfig.snowflake.database || '',
              username: envConfig.snowflake.username || '',
              account: envConfig.snowflake.account || '',
              warehouse: envConfig.snowflake.warehouse || '',
              schema: envConfig.snowflake.schema || 'PUBLIC',
              role: envConfig.snowflake.role || '',
              port: '443'
            }));
            return; // Exit after setting environment config
          }
        }
        
        // If no environment config, try localStorage
        console.log('💾 No environment config found, checking localStorage...');
        const savedConfig = localStorage.getItem('databaseConfig');
        if (savedConfig) {
          console.log('✅ Found saved config in localStorage');
          const config = JSON.parse(savedConfig);
          setDbConfig(config);
        } else {
          console.log('ℹ️ No saved configuration found, using defaults');
        }
        
      } catch (error) {
        console.error('❌ Failed to load configuration from backend:', error);
        
        // Fall back to localStorage only
        const savedConfig = localStorage.getItem('databaseConfig');
        if (savedConfig) {
          try {
            console.log('🔄 Falling back to localStorage config');
            const config = JSON.parse(savedConfig);
            setDbConfig(config);
          } catch (e) {
            console.error('❌ Failed to load database config from localStorage:', e);
          }
        }
      }
    };

    loadConfiguration();
  }, []);

  const handleConfigChange = (field: keyof DatabaseConfig, value: string) => {
    setDbConfig(prev => ({
      ...prev,
      [field]: value
    }));
    setConnectionStatus('idle');
  };

  const handleTypeChange = (type: DatabaseConfig['type']) => {
    const defaultPorts = {
      snowflake: '443',
      postgres: '5432',
      mysql: '3306',
      azure_sql: '1433'
    };

    setDbConfig(prev => ({
      ...prev,
      type,
      port: defaultPorts[type],
      // Clear type-specific fields when switching
      account: type === 'snowflake' ? prev.account : undefined,
      warehouse: type === 'snowflake' ? prev.warehouse : undefined,
      schema: type === 'snowflake' ? prev.schema : undefined,
      role: type === 'snowflake' ? prev.role : undefined,
      driver: type === 'azure_sql' ? 'ODBC Driver 17 for SQL Server' : undefined
    }));
    setConnectionStatus('idle');
  };

  const testConnection = async () => {
    setIsConnecting(true);
    setConnectionStatus('idle');
    setStatusMessage('');

    try {
      const requestData = {
        dbType: dbConfig.type,
        config: {
          host: dbConfig.host,
          port: dbConfig.port,
          database: dbConfig.database,
          username: dbConfig.username,
          password: dbConfig.password,
          // Include Snowflake-specific fields
          account: dbConfig.account,
          warehouse: dbConfig.warehouse,
          schema: dbConfig.schema,
          role: dbConfig.role
        }
      };

      const response = await axios.post('http://localhost:8003/test-connection', requestData);
      if (response.data.success) {
        setConnectionStatus('success');
        setStatusMessage('Connection successful!');
        // Save config to localStorage
        localStorage.setItem('databaseConfig', JSON.stringify(dbConfig));
        // Notify parent component
        onDatabaseChange(dbConfig.type);
      } else {
        setConnectionStatus('error');
        setStatusMessage(response.data.error || 'Connection failed');
      }
    } catch (error: any) {
      setConnectionStatus('error');
      setStatusMessage(error.response?.data?.error || error.message || 'Connection failed');
    } finally {
      setIsConnecting(false);
    }
  };

  const saveConfig = () => {
    localStorage.setItem('databaseConfig', JSON.stringify(dbConfig));
    onDatabaseChange(dbConfig.type);
    setStatusMessage('Configuration saved!');
    setTimeout(() => setStatusMessage(''), 3000);
  };

  if (!isOpen) return null;

  return (
    <div style={{
      position: 'fixed',
      top: 0,
      left: 0,
      right: 0,
      bottom: 0,
      backgroundColor: 'rgba(0, 0, 0, 0.5)',
      display: 'flex',
      alignItems: 'center',
      justifyContent: 'center',
      zIndex: 1000
    }}>
      <div style={{
        backgroundColor: 'white',
        borderRadius: '12px',
        padding: '1.5rem',
        width: '90%',
        maxWidth: '800px',
        maxHeight: '90vh',
        overflow: 'auto',
        boxShadow: '0 20px 25px -5px rgba(0, 0, 0, 0.1), 0 10px 10px -5px rgba(0, 0, 0, 0.04)'
      }}>
        {/* Header */}
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1.5rem' }}>
          <h2 style={{ margin: 0, color: '#374151', fontSize: '1.5rem', fontWeight: '600' }}>Settings</h2>
          <button
            onClick={onClose}
            style={{
              background: 'none',
              border: 'none',
              fontSize: '1.5rem',
              cursor: 'pointer',
              color: '#9ca3af',
              padding: '0.25rem'
            }}
          >
            ✕
          </button>
        </div>

        {/* Tabs */}
        <div style={{ display: 'flex', marginBottom: '2rem', borderBottom: '2px solid #f3f4f6' }}>
          <button
            onClick={() => setActiveTab('database')}
            style={{
              padding: '0.75rem 1.5rem',
              background: 'none',
              border: 'none',
              borderBottom: activeTab === 'database' ? '2px solid #3b82f6' : '2px solid transparent',
              color: activeTab === 'database' ? '#3b82f6' : '#6b7280',
              cursor: 'pointer',
              fontWeight: '500',
              fontSize: '1rem'
            }}
          >
            Database Connection
          </button>
          <button
            onClick={() => setActiveTab('general')}
            style={{
              padding: '0.75rem 1.5rem',
              background: 'none',
              border: 'none',
              borderBottom: activeTab === 'general' ? '2px solid #3b82f6' : '2px solid transparent',
              color: activeTab === 'general' ? '#3b82f6' : '#6b7280',
              cursor: 'pointer',
              fontWeight: '500',
              fontSize: '1rem'
            }}
          >
            General
          </button>
        </div>

        {/* Content */}
        {activeTab === 'database' && (
          <div>
            <h3 style={{ color: '#374151', fontSize: '1.125rem', fontWeight: '600', marginBottom: '1rem' }}>
              Configure Database Connection
            </h3>

            {/* Database Type Selection */}
            <div style={{ marginBottom: '1.5rem' }}>
              <label style={{ display: 'block', marginBottom: '0.5rem', color: '#374151', fontWeight: '500' }}>
                Database Type
              </label>
              <select
                value={dbConfig.type}
                onChange={(e) => handleTypeChange(e.target.value as DatabaseConfig['type'])}
                style={{
                  width: '100%',
                  padding: '0.75rem',
                  border: '2px solid #d1d5db',
                  borderRadius: '8px',
                  fontSize: '1rem',
                  backgroundColor: 'white'
                }}
              >
                <option value="snowflake">❄️ Snowflake</option>
                <option value="postgres">🐘 PostgreSQL</option>
                <option value="mysql">🐬 MySQL</option>
                <option value="azure_sql">☁️ Azure SQL</option>
              </select>
            </div>

            {/* Connection Form */}
            <div style={{ display: 'grid', gap: '1rem' }}>
              {/* Snowflake specific fields */}
              {dbConfig.type === 'snowflake' && (
                <>
                  <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '1rem' }}>
                    <div>
                      <label style={{ display: 'block', marginBottom: '0.5rem', color: '#374151', fontWeight: '500' }}>
                        Account
                      </label>
                      <input
                        type="text"
                        value={dbConfig.account || ''}
                        onChange={(e) => handleConfigChange('account', e.target.value)}
                        placeholder="your-account.snowflakecomputing.com"
                        style={{
                          width: '100%',
                          padding: '0.75rem',
                          border: '2px solid #d1d5db',
                          borderRadius: '8px',
                          fontSize: '1rem'
                        }}
                      />
                    </div>
                    <div>
                      <label style={{ display: 'block', marginBottom: '0.5rem', color: '#374151', fontWeight: '500' }}>
                        Port
                      </label>
                      <input
                        type="text"
                        value={dbConfig.port}
                        onChange={(e) => handleConfigChange('port', e.target.value)}
                        placeholder="443"
                        style={{
                          width: '100%',
                          padding: '0.75rem',
                          border: '2px solid #d1d5db',
                          borderRadius: '8px',
                          fontSize: '1rem'
                        }}
                      />
                    </div>
                  </div>
                  <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '1rem' }}>
                    <div>
                      <label style={{ display: 'block', marginBottom: '0.5rem', color: '#374151', fontWeight: '500' }}>
                        Database
                      </label>
                      <input
                        type="text"
                        value={dbConfig.database}
                        onChange={(e) => handleConfigChange('database', e.target.value)}
                        placeholder="AZURE_ANALYTICS"
                        style={{
                          width: '100%',
                          padding: '0.75rem',
                          border: '2px solid #d1d5db',
                          borderRadius: '8px',
                          fontSize: '1rem'
                        }}
                      />
                    </div>
                    <div>
                      <label style={{ display: 'block', marginBottom: '0.5rem', color: '#374151', fontWeight: '500' }}>
                        Username
                      </label>
                      <input
                        type="text"
                        value={dbConfig.username}
                        onChange={(e) => handleConfigChange('username', e.target.value)}
                        placeholder="your_username"
                        style={{
                          width: '100%',
                          padding: '0.75rem',
                          border: '2px solid #d1d5db',
                          borderRadius: '8px',
                          fontSize: '1rem'
                        }}
                      />
                    </div>
                  </div>
                  <div>
                    <label style={{ display: 'block', marginBottom: '0.5rem', color: '#374151', fontWeight: '500' }}>
                      Password
                    </label>
                    <input
                      type="password"
                      value={dbConfig.password}
                      onChange={(e) => handleConfigChange('password', e.target.value)}
                      placeholder="your_password"
                      style={{
                        width: '100%',
                        padding: '0.75rem',
                        border: '2px solid #d1d5db',
                        borderRadius: '8px',
                        fontSize: '1rem'
                      }}
                    />
                  </div>
                  <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', gap: '1rem' }}>
                    <div>
                      <label style={{ display: 'block', marginBottom: '0.5rem', color: '#374151', fontWeight: '500' }}>
                        Warehouse
                      </label>
                      <input
                        type="text"
                        value={dbConfig.warehouse || ''}
                        onChange={(e) => handleConfigChange('warehouse', e.target.value)}
                        placeholder="COMPUTE_WH"
                        style={{
                          width: '100%',
                          padding: '0.75rem',
                          border: '2px solid #d1d5db',
                          borderRadius: '8px',
                          fontSize: '1rem'
                        }}
                      />
                    </div>
                    <div>
                      <label style={{ display: 'block', marginBottom: '0.5rem', color: '#374151', fontWeight: '500' }}>
                        Schema
                      </label>
                      <input
                        type="text"
                        value={dbConfig.schema || ''}
                        onChange={(e) => handleConfigChange('schema', e.target.value)}
                        placeholder="PUBLIC"
                        style={{
                          width: '100%',
                          padding: '0.75rem',
                          border: '2px solid #d1d5db',
                          borderRadius: '8px',
                          fontSize: '1rem'
                        }}
                      />
                    </div>
                    <div>
                      <label style={{ display: 'block', marginBottom: '0.5rem', color: '#374151', fontWeight: '500' }}>
                        Role (optional)
                      </label>
                      <input
                        type="text"
                        value={dbConfig.role || ''}
                        onChange={(e) => handleConfigChange('role', e.target.value)}
                        placeholder="ACCOUNTADMIN"
                        style={{
                          width: '100%',
                          padding: '0.75rem',
                          border: '2px solid #d1d5db',
                          borderRadius: '8px',
                          fontSize: '1rem'
                        }}
                      />
                    </div>
                  </div>
                </>
              )}

              {/* Standard fields for other database types */}
              {dbConfig.type !== 'snowflake' && (
                <>
                  <div style={{ display: 'grid', gridTemplateColumns: '2fr 1fr', gap: '1rem' }}>
                    <div>
                      <label style={{ display: 'block', marginBottom: '0.5rem', color: '#374151', fontWeight: '500' }}>
                        Host
                      </label>
                      <input
                        type="text"
                        value={dbConfig.host}
                        onChange={(e) => handleConfigChange('host', e.target.value)}
                        placeholder="localhost"
                        style={{
                          width: '100%',
                          padding: '0.75rem',
                          border: '2px solid #d1d5db',
                          borderRadius: '8px',
                          fontSize: '1rem'
                        }}
                      />
                    </div>
                    <div>
                      <label style={{ display: 'block', marginBottom: '0.5rem', color: '#374151', fontWeight: '500' }}>
                        Port
                      </label>
                      <input
                        type="text"
                        value={dbConfig.port}
                        onChange={(e) => handleConfigChange('port', e.target.value)}
                        style={{
                          width: '100%',
                          padding: '0.75rem',
                          border: '2px solid #d1d5db',
                          borderRadius: '8px',
                          fontSize: '1rem'
                        }}
                      />
                    </div>
                  </div>
                  <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '1rem' }}>
                    <div>
                      <label style={{ display: 'block', marginBottom: '0.5rem', color: '#374151', fontWeight: '500' }}>
                        Database
                      </label>
                      <input
                        type="text"
                        value={dbConfig.database}
                        onChange={(e) => handleConfigChange('database', e.target.value)}
                        style={{
                          width: '100%',
                          padding: '0.75rem',
                          border: '2px solid #d1d5db',
                          borderRadius: '8px',
                          fontSize: '1rem'
                        }}
                      />
                    </div>
                    <div>
                      <label style={{ display: 'block', marginBottom: '0.5rem', color: '#374151', fontWeight: '500' }}>
                        Username
                      </label>
                      <input
                        type="text"
                        value={dbConfig.username}
                        onChange={(e) => handleConfigChange('username', e.target.value)}
                        style={{
                          width: '100%',
                          padding: '0.75rem',
                          border: '2px solid #d1d5db',
                          borderRadius: '8px',
                          fontSize: '1rem'
                        }}
                      />
                    </div>
                  </div>
                  <div>
                    <label style={{ display: 'block', marginBottom: '0.5rem', color: '#374151', fontWeight: '500' }}>
                      Password
                    </label>
                    <input
                      type="password"
                      value={dbConfig.password}
                      onChange={(e) => handleConfigChange('password', e.target.value)}
                      style={{
                        width: '100%',
                        padding: '0.75rem',
                        border: '2px solid #d1d5db',
                        borderRadius: '8px',
                        fontSize: '1rem'
                      }}
                    />
                  </div>
                </>
              )}
            </div>

            {/* Action Buttons */}
            <div style={{ display: 'flex', gap: '1rem', marginTop: '2rem', justifyContent: 'flex-end' }}>
              <button
                onClick={testConnection}
                disabled={isConnecting}
                style={{
                  padding: '0.75rem 1.5rem',
                  backgroundColor: '#3b82f6',
                  color: 'white',
                  border: 'none',
                  borderRadius: '8px',
                  cursor: isConnecting ? 'not-allowed' : 'pointer',
                  fontWeight: '500',
                  fontSize: '1rem',
                  opacity: isConnecting ? 0.6 : 1
                }}
              >
                {isConnecting ? 'Testing...' : 'Test Connection'}
              </button>
              <button
                onClick={saveConfig}
                style={{
                  padding: '0.75rem 1.5rem',
                  backgroundColor: '#10b981',
                  color: 'white',
                  border: 'none',
                  borderRadius: '8px',
                  cursor: 'pointer',
                  fontWeight: '500',
                  fontSize: '1rem'
                }}
              >
                Save Configuration
              </button>
            </div>

            {/* Status Message */}
            {statusMessage && (
              <div style={{
                marginTop: '1rem',
                padding: '0.75rem',
                borderRadius: '8px',
                backgroundColor: connectionStatus === 'success' ? '#d1fae5' : connectionStatus === 'error' ? '#fee2e2' : '#f3f4f6',
                color: connectionStatus === 'success' ? '#065f46' : connectionStatus === 'error' ? '#991b1b' : '#374151',
                border: `1px solid ${connectionStatus === 'success' ? '#10b981' : connectionStatus === 'error' ? '#ef4444' : '#d1d5db'}`
              }}>
                {statusMessage}
              </div>
            )}
          </div>
        )}

        {activeTab === 'general' && (
          <div>
            <h3 style={{ color: '#374151', fontSize: '1.125rem', fontWeight: '600', marginBottom: '1rem' }}>
              General Settings
            </h3>
            <p style={{ color: '#6b7280' }}>General settings will be available in future versions.</p>
          </div>
        )}
      </div>
    </div>
  );
}

export default SettingsPanel;
