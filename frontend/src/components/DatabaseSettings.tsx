import React, { useState, useEffect, useCallback, useMemo } from 'react';
import './DatabaseSettings.css';

interface DatabaseSettingsProps {
  onNavigateBack: () => void;
}

interface IndexingStatus {
  isIndexed: boolean;
  isIndexing: boolean;
  totalTables: number;
  indexedTables: number;
  indexedTableNames?: string[];  // List of actual indexed table names
  lastIndexed: string | null;
  error: string | null;
}

interface TestResult {
  success: boolean;
  message: string;
  latency_ms?: number;
  database_type?: string;
  database_name?: string;
}

const DatabaseSettings: React.FC<DatabaseSettingsProps> = ({ onNavigateBack }) => {
  const [config, setConfig] = useState({
    database_type: 'azure_sql',
    host: '',
    port: '1433',
    database: '',
    username: '',
    password: ''
  });

  const [testResult, setTestResult] = useState<TestResult | null>(null);
  const [isTestingConnection, setIsTestingConnection] = useState(false);
  const [isSaving, setIsSaving] = useState(false);
  const [isRefreshingStatus, setIsRefreshingStatus] = useState(false);
  const [showScrollTop, setShowScrollTop] = useState(false);

  const [indexingStatus, setIndexingStatus] = useState<IndexingStatus>({
    isIndexed: false,
    isIndexing: false,
    totalTables: 0,
    indexedTables: 0,
    indexedTableNames: [],
    lastIndexed: null,
    error: null
  });

  // Database status bar state
  const [dbStatus, setDbStatus] = useState<any>(null);
  const [isLoadingDbStatus, setIsLoadingDbStatus] = useState(true);

  // Smart table selection state
  const [tableInput, setTableInput] = useState('');
  const [selectedTables, setSelectedTables] = useState<string[]>([]);
  const [suggestions, setSuggestions] = useState<any[]>([]);
  const [showSuggestions, setShowSuggestions] = useState(false);
  const [isIndexingTables, setIsIndexingTables] = useState(false);

  // Table caching system
  const [allTablesCache, setAllTablesCache] = useState<any[]>([]);
  const [isLoadingTables, setIsLoadingTables] = useState(false);
  const [tablesCacheTime, setTablesCacheTime] = useState<number | null>(null);
  const [cacheStatus, setCacheStatus] = useState<'loading' | 'loaded' | 'error' | 'empty'>('empty');
  const [cacheDetails, setCacheDetails] = useState<string>('');

  // Load saved configuration and database status
  useEffect(() => {
    const initializeSettings = async () => {
      await loadSavedConfig();
      await loadDatabaseStatus(); // This will load backend connection status and cache tables
      await checkIndexingStatus();
    };
    initializeSettings();
  }, []);

  // Refresh cache when database connection changes
  useEffect(() => {
    if (dbStatus?.connected && allTablesCache.length === 0 && cacheStatus !== 'loading') {
      console.log('Database connected, preloading tables...');
      preloadAllTables();
    }
  }, [dbStatus?.connected]);

  // Debug logging
  useEffect(() => {
    console.log('Cache status:', cacheStatus, 'Tables cached:', allTablesCache.length);
  }, [cacheStatus, allTablesCache.length]);

  const loadDatabaseStatus = async () => {
    setIsLoadingDbStatus(true);
    console.log('Loading database status...');
    
    try {
      // Add timeout to prevent hanging
      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), 10000); // 10 second timeout
      
      const response = await fetch('/api/database/status', {
        signal: controller.signal
      });
      clearTimeout(timeoutId);
      console.log('Database status response:', response.status);
      
      if (response.ok) {
        const status = await response.json();
        console.log('Database status data:', status);
        
        // Map backend field names to frontend expectations
        const mappedStatus = {
          connected: status.isConnected || false,
          database_type: status.databaseType || 'Unknown',
          database_name: status.database || 'Unknown',
          server: status.server || 'Unknown',
          schema: status.schema || '',
          warehouse: status.warehouse || '',
          lastConnected: status.lastConnected || null
        };
        
        console.log('Mapped status:', mappedStatus);
        setDbStatus(mappedStatus);
        
        // Update config to reflect currently connected database
        if (status.isConnected) {
          updateConfigWithConnectedDatabase(status);
          await preloadAllTables(); // Load table names from backend cache
        }
      } else {
        console.error('Database status API error:', response.status, response.statusText);
        const errorText = await response.text();
        console.error('Error response:', errorText);
        
        setDbStatus({
          connected: false,
          database_type: 'API Error',
          database_name: `HTTP ${response.status}: ${response.statusText}`
        });
      }
    } catch (error) {
      console.error('Failed to load database status:', error);
      
      let errorMessage = 'Unknown error';
      let errorType = 'Network Error';
      
      if (error instanceof Error) {
        if (error.name === 'AbortError') {
          errorType = 'Timeout Error';
          errorMessage = 'Request timed out after 10 seconds';
        } else {
          errorMessage = error.message;
        }
      }
      
      setDbStatus({
        connected: false,
        database_type: errorType,
        database_name: errorMessage
      });
    } finally {
      console.log('Database status loading complete');
      setIsLoadingDbStatus(false);
    }
  };

  // Update config form to show currently connected database
  const updateConfigWithConnectedDatabase = (status: any) => {
    const databaseTypeMapping: { [key: string]: string } = {
      'Azure SQL': 'azure_sql',
      'Snowflake': 'snowflake',
      'PostgreSQL': 'postgres',
      'SQLite': 'sqlite'
    };

    const portMapping: { [key: string]: string } = {
      'azure_sql': '1433',
      'snowflake': '443',
      'postgres': '5432',
      'sqlite': ''
    };

    const mappedType = databaseTypeMapping[status.databaseType] || 'azure_sql';
    
    setConfig(prevConfig => ({
      ...prevConfig,
      database_type: mappedType,
      host: status.server || prevConfig.host,
      port: portMapping[mappedType] || prevConfig.port,
      database: status.database || prevConfig.database,
      // Keep existing username/password from saved config for security
      // but update the display fields for currently connected database
    }));
  };

  // Preload all tables for instant search
  const preloadAllTables = async () => {
    // Check if we have recent cache (less than 5 minutes old)
    const now = Date.now();
    if (tablesCacheTime && (now - tablesCacheTime) < 300000 && allTablesCache.length > 0) {
      console.log('Using cached tables data');
      return;
    }

    setIsLoadingTables(true);
    setCacheStatus('loading');
    
    try {
      // Use the fast table names endpoint first for instant search
      const response = await fetch('/api/database/table-names');
      if (response.ok) {
        const data = await response.json();
        const tables = data.tables || [];
        
        // Simple table data for fast search - just names
        const simpleTables = tables.map((table: any) => ({
          name: table.name || table.full_name || table,
          schema: table.schema || 'dbo',
          full_name: table.full_name || table.name || table,
          searchText: `${table.name || table} ${table.schema || ''}`.toLowerCase(),
          priority: 'medium', // Default priority
          cached: true
        }));

        setAllTablesCache(simpleTables);
        setTablesCacheTime(now);
        setCacheStatus('loaded');
        setCacheDetails(`${simpleTables.length} tables cached (${Math.round((data.cache_age_seconds || 0))}s old)`);
        
        console.log(`✅ Loaded ${simpleTables.length} table names from backend cache for instant search`);
      } else {
        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
      }
    } catch (error) {
      console.error('Failed to preload tables:', error);
      setCacheStatus('error');
      setAllTablesCache([]);
    } finally {
      setIsLoadingTables(false);
    }
  };

  // Scroll to top button visibility
  useEffect(() => {
    const handleScroll = () => {
      setShowScrollTop(window.scrollY > 300);
    };
    window.addEventListener('scroll', handleScroll);
    return () => window.removeEventListener('scroll', handleScroll);
  }, []);

  const scrollToTop = () => {
    window.scrollTo({ top: 0, behavior: 'smooth' });
  };

  const loadSavedConfig = async () => {
    try {
      const response = await fetch('/api/database/config');
      if (response.ok) {
        const savedConfig = await response.json();
        setConfig({
          database_type: savedConfig.database_type || 'azure_sql',
          host: savedConfig.host || '',
          port: savedConfig.port || '1433',
          database: savedConfig.database || '',
          username: savedConfig.username || '',
          password: savedConfig.password || ''
        });
      }
    } catch (error) {
      console.error('Failed to load saved config:', error);
    }
  };

  const checkIndexingStatus = async () => {
    setIsRefreshingStatus(true);
    try {
      const response = await fetch('/api/database/indexing-status');
      if (response.ok) {
        const status = await response.json();
        setIndexingStatus({
          isIndexed: status.isIndexed || false,
          isIndexing: status.isIndexing || false,
          totalTables: status.totalTables || 0,
          indexedTables: status.indexedTables || 0,
          indexedTableNames: status.indexedTableNames || [],
          lastIndexed: status.lastIndexed || null,
          error: status.error || null
        });
      }
    } catch (error) {
      console.error('Failed to check indexing status:', error);
    } finally {
      setIsRefreshingStatus(false);
    }
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

      const result = await response.json();
      setTestResult(result);
    } catch (error) {
      setTestResult({
        success: false,
        message: `Connection failed: ${error}`
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
        alert('Configuration saved successfully!');
      } else {
        alert('Failed to save configuration');
      }
    } catch (error) {
      alert(`Error saving configuration: ${error}`);
    } finally {
      setIsSaving(false);
    }
  };

  // Enterprise-grade debounced table search
  const [isSearching, setIsSearching] = useState(false);

  // Clear searching state when component loads
  useEffect(() => {
    setIsSearching(false);
  }, []);

  // Fast debounced search function (reduced delay since we're using cache)
  const debouncedSearch = useMemo(() => {
    let timeoutId: NodeJS.Timeout;
    return (value: string) => {
      clearTimeout(timeoutId);
      // Immediate search if cache is loaded, debounced if not
      const delay = cacheStatus === 'loaded' ? 50 : 150;
      timeoutId = setTimeout(() => {
        performTableSearch(value);
      }, delay);
    };
  }, [allTablesCache, cacheStatus]);

  const performTableSearch = (value: string) => {
    if (!value) {
      setShowSuggestions(false);
      setIsSearching(false);
      return;
    }

    // If tables aren't cached yet, don't search and show appropriate message
    if (allTablesCache.length === 0) {
      if (cacheStatus === 'loading') {
        // Show loading state but don't set isSearching
        setSuggestions([]);
        setShowSuggestions(false);
        return;
      } else if (cacheStatus === 'error') {
        // Show error state
        setSuggestions([]);
        setShowSuggestions(false);
        return;
      } else {
        // Try to load tables if not already loading
        preloadAllTables();
        setSuggestions([]);
        setShowSuggestions(false);
        return;
      }
    }

    // Instant search through cached tables (no loading state needed)
    try {
      const searchTerm = value.toLowerCase();
      
      // Fast filtering through cached data
      const filtered = allTablesCache
        .filter((table: any) => 
          table.searchText?.includes(searchTerm) ||
          table.name?.toLowerCase().includes(searchTerm)
        )
        .map((table: any) => ({
          ...table,
          matchScore: calculateMatchScore(table, value)
        }))
        .sort((a: any, b: any) => b.matchScore - a.matchScore)
        .slice(0, 8);

      setSuggestions(filtered);
      setShowSuggestions(true);
      setIsSearching(false); // Ensure searching state is cleared
    } catch (error) {
      console.error('Error during table search:', error);
      setSuggestions([]);
      setShowSuggestions(false);
      setIsSearching(false);
    }
  };

  const calculateMatchScore = (table: any, searchTerm: string): number => {
    const term = searchTerm.toLowerCase();
    const name = table.name.toLowerCase();
    let score = 0;

    // Exact name match gets highest score
    if (name === term) score += 100;
    // Name starts with term
    else if (name.startsWith(term)) score += 80;
    // Name contains term
    else if (name.includes(term)) score += 60;
    
    // Priority boost
    if (table.priority === 'high') score += 20;
    else if (table.priority === 'medium') score += 10;
    
    // Row count boost (more data = more relevant)
    if (table.row_count > 1000) score += 15;
    else if (table.row_count > 100) score += 10;
    
    return score;
  };

  // Handle table input change with debouncing
  const handleTableInputChange = (value: string) => {
    setTableInput(value);
    debouncedSearch(value);
  };

  // Add table to selection
  const addTable = (tableName: string) => {
    if (!selectedTables.includes(tableName)) {
      setSelectedTables([...selectedTables, tableName]);
    }
    setTableInput('');
    setShowSuggestions(false);
  };

  // Remove table from selection
  const removeTable = (tableName: string) => {
    setSelectedTables(selectedTables.filter(t => t !== tableName));
  };

  // Enterprise indexing progress tracking
  const [indexingProgress, setIndexingProgress] = useState({
    current: 0,
    total: 0,
    currentTable: '',
    eta: '',
    completedTables: [] as string[],
    errors: [] as string[]
  });

  // Start indexing with real-time progress
  const startTableIndexing = async (forceReindex = false) => {
    if (selectedTables.length === 0) return;
    
    // If force reindex, show confirmation
    if (forceReindex) {
      const confirmed = window.confirm(
        `⚠️ Force Reindex will COMPLETELY CLEAR the existing vector index and rebuild it from scratch with the selected ${selectedTables.length} table${selectedTables.length !== 1 ? 's' : ''}.\n\n` +
        `This is recommended when:\n` +
        `• Switching to a different database\n` +
        `• Starting fresh with new table selection\n` +
        `• Fixing index corruption issues\n\n` +
        `Continue with Force Reindex?`
      );
      if (!confirmed) return;
    }
    
    setIsIndexingTables(true);
    setIndexingProgress({
      current: 0,
      total: selectedTables.length,
      currentTable: selectedTables[0],
      eta: 'Calculating...',
      completedTables: [],
      errors: []
    });

    try {
      const response = await fetch('/api/database/start-indexing', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          force_reindex: forceReindex,
          selected_tables: selectedTables
        })
      });

      if (response.ok) {
        const result = await response.json();
        console.log('Indexing started:', result);
        
        // Start progress monitoring
        startProgressMonitoring();
        
        await checkIndexingStatus();
      } else {
        const error = await response.text();
        setIndexingProgress(prev => ({
          ...prev,
          errors: [...prev.errors, `Failed to start indexing: ${error}`]
        }));
      }
    } catch (error) {
      console.error('Error starting indexing:', error);
      setIndexingProgress(prev => ({
        ...prev,
        errors: [...prev.errors, `Network error: ${error}`]
      }));
    } finally {
      setIsIndexingTables(false);
    }
  };

  // Real-time progress monitoring
  const startProgressMonitoring = () => {
    const interval = setInterval(async () => {
      try {
        const response = await fetch('/api/database/indexing-progress');
        if (response.ok) {
          const progress = await response.json();
          setIndexingProgress(prev => ({
            ...prev,
            current: progress.completed || prev.current,
            currentTable: progress.currentTable || prev.currentTable,
            eta: progress.eta || prev.eta,
            completedTables: progress.completedTables || prev.completedTables,
            errors: progress.errors || prev.errors
          }));

          // Stop monitoring when complete
          if (progress.completed >= selectedTables.length) {
            clearInterval(interval);
            setIsIndexingTables(false);
          }
        }
      } catch (error) {
        console.error('Progress monitoring error:', error);
      }
    }, 2000);

    // Cleanup after 10 minutes max
    setTimeout(() => clearInterval(interval), 600000);
  };

  // Enhanced Database Status Bar Component
  const renderDatabaseStatusBar = () => {
    const getStatusIcon = () => {
      if (isLoadingDbStatus) return '⏳';
      return dbStatus?.connected ? '🟢' : '🔴';
    };

    const getStatusText = () => {
      if (isLoadingDbStatus) return 'Checking connection...';
      return dbStatus?.connected ? 'Connected' : 'Not Connected';
    };

    const renderConnectionDetails = () => {
      if (isLoadingDbStatus) return 'Loading database information...';
      
      if (!dbStatus?.connected) {
        return 'Database connection not available';
      }

      return (
        <div className="connection-details">
          <div className="primary-details">
            <span className="db-type">{dbStatus.database_type}</span>
            <span className="separator">•</span>
            <span className="db-name">{dbStatus.database_name}</span>
          </div>
          <div className="secondary-details">
            <span className="server-info">📡 {dbStatus.server}</span>
            {dbStatus.schema && <span className="schema-info">🗂️ Schema: {dbStatus.schema}</span>}
            {dbStatus.lastConnected && (
              <span className="last-connected">
                🕒 Connected: {new Date(dbStatus.lastConnected * 1000).toLocaleTimeString()}
              </span>
            )}
          </div>
        </div>
      );
    };

    return (
      <div className="database-status-bar">
        <div className="status-indicator">
          <span className={`status-icon ${isLoadingDbStatus ? 'loading' : ''}`}>
            {getStatusIcon()}
          </span>
          <div className="status-details">
            <div className={`status-title ${dbStatus?.connected ? 'connected' : 'disconnected'}`}>
              {getStatusText()}
            </div>
            <div className="status-subtitle">
              {renderConnectionDetails()}
            </div>
          </div>
        </div>
        <div className="status-actions">
          <button 
            onClick={loadDatabaseStatus} 
            className={`refresh-status-btn ${isLoadingDbStatus ? 'loading' : ''}`}
            disabled={isLoadingDbStatus}
            title="Refresh connection status"
          >
            🔄 {isLoadingDbStatus ? 'Refreshing...' : 'Refresh'}
          </button>
        </div>
      </div>
    );
  };

  // Currently Indexed Tables Display Component
  const renderIndexedTablesStatus = () => {
    return (
      <div className="indexed-tables-status">
        <div className="subsection-header">
          <span>✅</span>
          <span>Currently Indexed Tables</span>
          <div className="subsection-info">
            <span className="subsection-subtitle">Tables available for AI-powered natural language queries</span>
          </div>
        </div>
        
        {indexingStatus.isIndexing ? (
          <div className="indexing-in-progress">
            <span className="info-text">⏳ Indexing in progress... Status will update when complete</span>
          </div>
        ) : indexingStatus.indexedTables > 0 ? (
          <div className="indexed-tables-display">
            <div className="indexed-count-banner">
              <span className="count-badge">{indexingStatus.indexedTables}</span>
              <span className="count-text">table{indexingStatus.indexedTables !== 1 ? 's' : ''} indexed and ready for AI queries</span>
            </div>
            
            {indexingStatus.indexedTableNames && indexingStatus.indexedTableNames.length > 0 && (
              <div className="indexed-tables-list">
                <div className="indexed-tables-grid">
                  {indexingStatus.indexedTableNames.map((tableName) => (
                    <div key={tableName} className="indexed-table-card">
                      <span className="table-icon">🗂️</span>
                      <span className="table-name">{tableName}</span>
                      <span className="indexed-badge">✅ Indexed</span>
                    </div>
                  ))}
                </div>
              </div>
            )}
            
            <div className="indexed-status-footer">
              <span className="last-indexed">
                {indexingStatus.lastIndexed ? 
                  `Last indexed: ${new Date(indexingStatus.lastIndexed).toLocaleString()}` : 
                  'Recently indexed'
                }
              </span>
              <button 
                onClick={checkIndexingStatus} 
                className="refresh-index-status-btn"
                title="Refresh indexed tables status"
              >
                🔄 Refresh Status
              </button>
            </div>
          </div>
        ) : (
          <div className="no-indexed-tables">
            <div className="empty-state">
              <span className="empty-icon">📋</span>
              <span className="empty-title">No tables indexed yet</span>
              <span className="empty-subtitle">Select and index tables below to enable AI-powered queries</span>
            </div>
          </div>
        )}
      </div>
    );
  };

  // Enhanced Smart Table Selection Component
  const renderSmartTableSelection = () => {
    return (
      <div className="smart-table-selection">
        <div className="subsection-header">
          <span>📋</span>
          <span>Select Tables for Reindexing</span>
          <div className="subsection-info">
            <span className="subsection-subtitle">Choose specific tables to reindex for enhanced AI analysis</span>
            <div className="cache-status">
              {cacheStatus === 'loading' && (
                <span className="cache-indicator loading">⏳ Loading tables...</span>
              )}
              {cacheStatus === 'loaded' && (
                <span className="cache-indicator loaded">
                  ⚡ {cacheDetails || `${allTablesCache.length} tables cached (instant search)`}
                </span>
              )}
              {cacheStatus === 'error' && (
                <span className="cache-indicator error">
                  ⚠️ Failed to load tables
                  <button onClick={preloadAllTables} className="retry-cache-btn">Retry</button>
                </span>
              )}
            </div>
          </div>
        </div>

        {/* Enterprise Table Search */}
        <div className="table-search-container">
          <div className="search-input-wrapper">
            <div className="search-icon">🔍</div>
            <input
              type="text"
              placeholder={
                cacheStatus === 'loaded' 
                  ? `Instant search through ${allTablesCache.length} tables...`
                  : cacheStatus === 'loading'
                  ? "Loading tables for search..."
                  : "Start typing to search available tables..."
              }
              value={tableInput}
              onChange={(e) => handleTableInputChange(e.target.value)}
              className="enterprise-table-input"
              onFocus={() => tableInput && setShowSuggestions(true)}
              disabled={cacheStatus === 'loading'}
              onKeyDown={(e) => {
                if (e.key === 'Enter' && tableInput && suggestions.length > 0) {
                  const firstTable = suggestions[0];
                  const tableName = typeof firstTable === 'string' ? firstTable : (firstTable?.name || firstTable);
                  addTable(tableName);
                  e.preventDefault();
                }
                if (e.key === 'Escape') {
                  setShowSuggestions(false);
                  setTableInput('');
                  e.preventDefault();
                }
                if (e.key === 'ArrowDown' && showSuggestions && suggestions.length > 0) {
                  // Future: implement arrow key navigation
                  e.preventDefault();
                }
              }}
            />
            <div className="input-actions">
              {tableInput && (
                <button 
                  onClick={() => {setTableInput(''); setShowSuggestions(false);}}
                  className="clear-input-btn"
                  title="Clear search"
                >
                  ❌
                </button>
              )}
              <button 
                onClick={preloadAllTables}
                className={`refresh-cache-btn ${isLoadingTables ? 'loading' : ''}`}
                disabled={isLoadingTables}
                title="Refresh table cache"
              >
                {isLoadingTables ? '⏳' : '🔄'}
              </button>
            </div>
          </div>

          {/* Enhanced Suggestions Dropdown */}
          {showSuggestions && (
            <div className="enterprise-suggestions-dropdown">
              <div className="suggestions-header">
                <span>
                  {isSearching ? '🔍 Searching...' : `Available Tables (${suggestions.length})`}
                </span>
                <span className="suggestions-hint">Click to select • Enter for first • Esc to close</span>
              </div>
              
              {cacheStatus === 'loading' ? (
                <div className="loading-suggestions">
                  <div className="loading-spinner-container">
                    <div className="loading-spinner"></div>
                    <span>Loading table cache for instant search...</span>
                  </div>
                </div>
              ) : cacheStatus === 'error' ? (
                <div className="error-suggestions">
                  <div className="error-container">
                    <span>⚠️ Failed to load tables</span>
                    <button onClick={preloadAllTables} className="retry-btn">
                      🔄 Retry
                    </button>
                  </div>
                </div>
              ) : suggestions.length > 0 ? (
                <div className="suggestions-list">
                  {suggestions.map((table: any, index) => (
                    <div
                      key={table.name || table}
                      className={`enterprise-suggestion-item ${index === 0 ? 'highlighted' : ''}`}
                      onClick={() => addTable(typeof table === 'string' ? table : table.name)}
                    >
                      <div className="suggestion-main">
                        <span className="table-icon">📋</span>
                        <div className="table-details">
                          <span className="table-name">{typeof table === 'string' ? table : table.name}</span>
                          {typeof table === 'object' && table.row_count && (
                            <span className="table-metadata">
                              📊 {table.row_count.toLocaleString()} rows
                              {table.priority && (
                                <span className={`priority-indicator ${table.priority}`}>
                                  {table.priority === 'high' ? '🔥' : table.priority === 'medium' ? '⭐' : '📋'}
                                </span>
                              )}
                            </span>
                          )}
                        </div>
                      </div>
                      {selectedTables.includes(typeof table === 'string' ? table : table.name) && (
                        <span className="already-selected">✅ Selected</span>
                      )}
                    </div>
                  ))}
                </div>
              ) : tableInput && cacheStatus === 'loaded' && (
                <div className="no-suggestions">
                  <span>🔍 No tables found matching "{tableInput}"</span>
                  <div className="search-stats">
                    Searched through {allTablesCache.length} cached tables
                  </div>
                  <button 
                    onClick={() => {setTableInput(''); setShowSuggestions(false);}}
                    className="clear-search-btn"
                  >
                    Clear Search
                  </button>
                </div>
              )}
            </div>
          )}

          {showSuggestions && suggestions.length === 0 && tableInput && (
            <div className="no-suggestions">
              <span>🔍 No tables found matching "{tableInput}"</span>
            </div>
          )}
        </div>

        {/* Selected Tables Display */}
        {selectedTables.length > 0 && (
          <div className="selected-tables-container">
            <div className="selected-tables-header">
              <span>✅ Selected Tables ({selectedTables.length})</span>
              <button
                onClick={() => setSelectedTables([])}
                className="clear-all-btn"
                disabled={isIndexingTables}
                title="Clear all selections"
              >
                🗑️ Clear All
              </button>
            </div>
            <div className="enterprise-table-tags">
              {selectedTables.map((tableName, index) => (
                <div key={tableName} className="enterprise-table-tag">
                  <span className="tag-number">{index + 1}</span>
                  <span className="tag-icon">📋</span>
                  <span className="tag-name">{tableName}</span>
                  <button 
                    onClick={() => removeTable(tableName)}
                    className="enterprise-remove-btn"
                    disabled={isIndexingTables}
                    title={`Remove ${tableName}`}
                  >
                    ❌
                  </button>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* Enterprise Action Panel */}
        <div className="enterprise-action-panel">
          {/* Progress Display */}
          {isIndexingTables && indexingProgress.total > 0 && (
            <div className="indexing-progress-display">
              <div className="progress-header">
                <span>⚡ Reindexing in Progress</span>
                <span className="progress-percentage">
                  {Math.round((indexingProgress.current / indexingProgress.total) * 100)}%
                </span>
              </div>
              
              <div className="progress-bar-container">
                <div className="progress-bar">
                  <div 
                    className="progress-fill"
                    style={{ width: `${(indexingProgress.current / indexingProgress.total) * 100}%` }}
                  ></div>
                </div>
                <div className="progress-details">
                  <span>Processing: {indexingProgress.currentTable}</span>
                  <span>ETA: {indexingProgress.eta}</span>
                </div>
              </div>

              {indexingProgress.completedTables.length > 0 && (
                <div className="completed-tables-summary">
                  <span>✅ Completed: {indexingProgress.completedTables.join(', ')}</span>
                </div>
              )}

              {indexingProgress.errors.length > 0 && (
                <div className="indexing-errors">
                  <span>⚠️ Errors: {indexingProgress.errors.length}</span>
                  <details>
                    <summary>View Details</summary>
                    {indexingProgress.errors.map((error, i) => (
                      <div key={i} className="error-detail">{error}</div>
                    ))}
                  </details>
                </div>
              )}
            </div>
          )}

          <div className="action-info">
            {selectedTables.length === 0 ? (
              <span className="info-text">💡 Select tables above to enable reindexing</span>
            ) : isIndexingTables ? (
              <span className="info-text">
                ⚡ Reindexing {indexingProgress.total} tables for enhanced AI analysis...
              </span>
            ) : indexingProgress.completedTables && indexingProgress.completedTables.length > 0 ? (
              <div className="completion-status">
                <span className="success-text">
                  ✅ Successfully indexed {indexingProgress.completedTables.length} table{indexingProgress.completedTables.length !== 1 ? 's' : ''} with smart schema analysis
                </span>
                <div className="completed-tables-list">
                  <strong>Indexed Tables:</strong>
                  {indexingProgress.completedTables.map((table, index) => (
                    <span key={table} className="completed-table-badge">
                      {table}
                    </span>
                  ))}
                </div>
              </div>
            ) : (
              <span className="info-text">
                🚀 Ready to reindex {selectedTables.length} table{selectedTables.length !== 1 ? 's' : ''} for enhanced AI analysis
              </span>
            )}
          </div>
          
          <div className="action-buttons-row">
            <button
              onClick={() => startTableIndexing(false)}
              disabled={selectedTables.length === 0 || isIndexingTables}
              className={`enterprise-start-btn ${selectedTables.length > 0 && !isIndexingTables ? 'ready' : 'disabled'}`}
            >
              <span className="btn-icon">
                {isIndexingTables ? '⏳' : selectedTables.length > 0 ? '➕' : '💡'}
              </span>
              <span className="btn-text">
                {isIndexingTables 
                  ? `Reindexing... (${indexingProgress.current}/${indexingProgress.total})`
                  : selectedTables.length > 0 
                    ? `Add ${selectedTables.length} Table${selectedTables.length !== 1 ? 's' : ''} to Index`
                    : 'Select Tables to Reindex'
                }
              </span>
            </button>

            <button
              onClick={() => startTableIndexing(true)}
              disabled={selectedTables.length === 0 || isIndexingTables}
              className={`enterprise-force-btn ${selectedTables.length > 0 && !isIndexingTables ? 'ready' : 'disabled'}`}
              title="Clear existing index and rebuild from scratch with selected tables"
            >
              <span className="btn-icon">
                {isIndexingTables ? '⏳' : selectedTables.length > 0 ? '🔄' : '⚠️'}
              </span>
              <span className="btn-text">
                {isIndexingTables 
                  ? `Force Reindexing...`
                  : selectedTables.length > 0 
                    ? `Force Reindex ${selectedTables.length} Table${selectedTables.length !== 1 ? 's' : ''}`
                    : 'Force Reindex (Clear & Rebuild)'
                }
              </span>
            </button>

            {isIndexingTables && (
              <button
                onClick={() => {
                  setIsIndexingTables(false);
                  setIndexingProgress(prev => ({ ...prev, errors: [...prev.errors, 'Reindexing cancelled by user'] }));
                }}
                className="cancel-indexing-btn"
              >
                ⏹️ Cancel Reindexing
              </button>
            )}
          </div>
        </div>
      </div>
    );
  };

  return (
    <div className="database-settings">
      <div className="database-settings-container">
        {/* Header */}
        <div className="database-settings-header">
          <div className="header-content">
            <h1 className="page-title">⚙️ Database Settings</h1>
            <p className="page-subtitle">Configure your database connection and indexing preferences</p>
          </div>
          <button onClick={onNavigateBack} className="back-button">
            ← Back to Chat
          </button>
        </div>

        <div className="settings-content">
          {/* Database Status Bar */}
          {renderDatabaseStatusBar()}

          {/* Connection Section */}
          <div className="connection-section">
            <div className="section-header-with-status">
              <h2 className="section-header">
                <span>🗄️</span>
                <span>Database Connection Configuration</span>
              </h2>
              {dbStatus?.connected && (
                <div className="connected-indicator">
                  <span className="indicator-dot"></span>
                  <span>Currently Connected: {dbStatus.database_type}</span>
                </div>
              )}
            </div>

            <div className="config-info">
              <p className="config-description">
                💡 Configure database connection settings below. The currently active connection is shown in the status bar above.
              </p>
            </div>

            <div className="form-group">
              <label>Database Type</label>
              <select
                value={config.database_type}
                onChange={(e) => setConfig({...config, database_type: e.target.value})}
                className="form-input"
              >
                <option value="azure_sql">Azure SQL Server</option>
                <option value="snowflake">Snowflake</option>
                <option value="postgres">PostgreSQL</option>
                <option value="sqlite">SQLite</option>
              </select>
            </div>

            <div className="form-row">
              <div className="form-group">
                <label>Host/Server</label>
                <input
                  type="text"
                  value={config.host}
                  onChange={(e) => setConfig({...config, host: e.target.value})}
                  className="form-input"
                  placeholder="server.database.windows.net"
                />
              </div>
              <div className="form-group">
                <label>Port</label>
                <input
                  type="text"
                  value={config.port}
                  onChange={(e) => setConfig({...config, port: e.target.value})}
                  className="form-input"
                  placeholder="1433"
                />
              </div>
            </div>

            <div className="form-group">
              <label>Database Name</label>
              <input
                type="text"
                value={config.database}
                onChange={(e) => setConfig({...config, database: e.target.value})}
                className="form-input"
                placeholder="your-database-name"
              />
            </div>

            <div className="form-row">
              <div className="form-group">
                <label>Username</label>
                <input
                  type="text"
                  value={config.username}
                  onChange={(e) => setConfig({...config, username: e.target.value})}
                  className="form-input"
                  placeholder="username"
                />
              </div>
              <div className="form-group">
                <label>Password</label>
                <input
                  type="password"
                  value={config.password}
                  onChange={(e) => setConfig({...config, password: e.target.value})}
                  className="form-input"
                  placeholder="password"
                />
              </div>
            </div>

            {/* Test Result */}
            {testResult && (
              <div className={`test-result ${testResult.success ? 'success' : 'error'}`}>
                <span className="result-icon">{testResult.success ? '✅' : '❌'}</span>
                <span className="result-message">{testResult.message}</span>
                {testResult.latency_ms && (
                  <span className="result-latency">({testResult.latency_ms}ms)</span>
                )}
              </div>
            )}

            {/* Action Buttons */}
            <div className="action-buttons">
              <button
                onClick={testConnection}
                disabled={isTestingConnection}
                className="action-button test-button"
              >
                {isTestingConnection && <div className="loading-spinner"></div>}
                <span>{isTestingConnection ? 'Testing...' : 'Test Connection'}</span>
              </button>

              <button
                onClick={saveConfiguration}
                disabled={isSaving}
                className="action-button save-button"
              >
                {isSaving && <div className="loading-spinner"></div>}
                <span>{isSaving ? 'Saving...' : 'Save Configuration'}</span>
              </button>
            </div>
          </div>

          {/* Smart Table Selection & Vector Search Indexing */}
          <div className="indexing-section">
            <h2 className="section-header">
              <span>🔍</span>
              <span>Vector Search & Table Reindexing</span>
            </h2>

            {/* Current Indexing Status */}
            <div className="indexing-status-container">
              <div className="status-row">
                <span className="status-label">Current Status:</span>
                <span className={`status-badge ${
                  indexingStatus?.isIndexing 
                    ? 'indexing'
                    : indexingStatus?.isIndexed
                    ? 'indexed'
                    : 'not-indexed'
                }`}>
                  {indexingStatus?.isIndexing 
                    ? '⏳ Indexing in Progress'
                    : indexingStatus?.isIndexed
                    ? '✅ Indexed'
                    : '⚪ Not Indexed'
                  }
                </span>
                <button 
                  onClick={checkIndexingStatus}
                  disabled={isRefreshingStatus}
                  className={`refresh-button ${isRefreshingStatus ? 'loading' : ''}`}
                >
                  🔄 {isRefreshingStatus ? 'Refreshing...' : 'Refresh'}
                </button>
              </div>
              
              {indexingStatus?.totalTables > 0 && (
                <div className="progress-info">
                  <span>📊 {indexingStatus.indexedTables} of {indexingStatus.totalTables} tables indexed</span>
                  {indexingStatus.lastIndexed && (
                    <span className="last-indexed">
                      🕒 Last indexed: {new Date(indexingStatus.lastIndexed).toLocaleString()}
                    </span>
                  )}
                </div>
              )}
            </div>

            {/* Currently Indexed Tables Status */}
            {renderIndexedTablesStatus()}

            {/* Smart Table Selection */}
            {renderSmartTableSelection()}
          </div>

          {/* Instructions */}
          <div className="instructions-section">
            <h3 className="instructions-header">
              <span>💡</span>
              <span>How it works</span>
            </h3>
            <div className="instructions-list">
              <p>1. Check database connection status at the top</p>
              <p>2. Select tables you want to index for AI analysis</p>
              <p>3. Start indexing to enable intelligent schema discovery</p>
              <p>4. Use natural language queries in the chat interface</p>
            </div>
          </div>
        </div>

        {/* Back to Top Button */}
        {showScrollTop && (
          <button
            onClick={scrollToTop}
            className="scroll-to-top-button"
            title="Back to top"
          >
            <span>↑</span>
          </button>
        )}
      </div>
    </div>
  );
};

export default DatabaseSettings;