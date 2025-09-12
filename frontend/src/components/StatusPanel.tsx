interface StatusPanelProps {
  healthData: any
}

export default function StatusPanel({ healthData }: StatusPanelProps) {
  if (!healthData) {
    return (
      <div className="animate-pulse">
        <div className="bg-white rounded-lg shadow-sm border p-6">
          <div className="h-4 bg-gray-200 rounded w-1/4 mb-4"></div>
          <div className="space-y-2">
            <div className="h-3 bg-gray-200 rounded"></div>
            <div className="h-3 bg-gray-200 rounded w-5/6"></div>
          </div>
        </div>
      </div>
    )
  }

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'healthy': return 'text-green-600 bg-green-100'
      case 'degraded': return 'text-yellow-600 bg-yellow-100'
      case 'unhealthy': return 'text-red-600 bg-red-100'
      default: return 'text-gray-600 bg-gray-100'
    }
  }

  return (
    <div className="space-y-6">
      {/* Overall Status */}
      <div className="bg-white rounded-lg shadow-sm border p-6">
        <div className="flex items-center justify-between mb-4">
          <h2 className="text-xl font-semibold text-gray-900">System Status</h2>
          <span className={`px-3 py-1 rounded-full text-sm font-medium ${getStatusColor(healthData.status)}`}>
            {healthData.status?.charAt(0).toUpperCase() + healthData.status?.slice(1)}
          </span>
        </div>
        
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4 text-sm">
          <div>
            <span className="text-gray-600">Version:</span>
            <span className="ml-2 font-medium">{healthData.version}</span>
          </div>
          <div>
            <span className="text-gray-600">Environment:</span>
            <span className="ml-2 font-medium">{healthData.environment}</span>
          </div>
          <div>
            <span className="text-gray-600">Last Updated:</span>
            <span className="ml-2 font-medium">
              {new Date(healthData.timestamp).toLocaleString()}
            </span>
          </div>
        </div>
      </div>

      {/* Services Status */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Database Service */}
        <div className="bg-white rounded-lg shadow-sm border p-6">
          <h3 className="text-lg font-medium text-gray-900 mb-4">Database Service</h3>
          
          <div className="space-y-3">
            <div className="flex items-center justify-between">
              <span className="text-sm text-gray-600">Overall Status:</span>
              <span className={`px-2 py-1 rounded text-xs font-medium ${
                getStatusColor(healthData.services?.database?.status)
              }`}>
                {healthData.services?.database?.status}
              </span>
            </div>
            
            <div className="border-t pt-3">
              <p className="text-sm text-gray-600 mb-2">Connections:</p>
              {healthData.services?.database?.connections && 
                Object.entries(healthData.services.database.connections).map(([db, status]: [string, any]) => (
                  <div key={db} className="flex items-center justify-between text-sm">
                    <span className="text-gray-700">{db}</span>
                    <span className={`w-2 h-2 rounded-full ${status ? 'bg-green-400' : 'bg-red-400'}`}></span>
                  </div>
                ))}
            </div>
            
            <div className="text-sm text-gray-500">
              Available: {healthData.services?.database?.available_databases?.length || 0} databases
            </div>
          </div>
        </div>

        {/* LLM Service */}
        <div className="bg-white rounded-lg shadow-sm border p-6">
          <h3 className="text-lg font-medium text-gray-900 mb-4">LLM Service</h3>
          
          <div className="space-y-3">
            <div className="flex items-center justify-between">
              <span className="text-sm text-gray-600">Overall Status:</span>
              <span className={`px-2 py-1 rounded text-xs font-medium ${
                getStatusColor(healthData.services?.llm?.status)
              }`}>
                {healthData.services?.llm?.status}
              </span>
            </div>
            
            <div className="border-t pt-3">
              <p className="text-sm text-gray-600 mb-2">Providers:</p>
              {healthData.services?.llm?.providers && 
                Object.entries(healthData.services.llm.providers).map(([provider, status]: [string, any]) => (
                  <div key={provider} className="flex items-center justify-between text-sm">
                    <span className="text-gray-700 capitalize">{provider}</span>
                    <span className={`w-2 h-2 rounded-full ${status ? 'bg-green-400' : 'bg-red-400'}`}></span>
                  </div>
                ))}
            </div>
            
            <div className="text-sm text-gray-500">
              Default: {healthData.services?.llm?.default_provider}
            </div>
          </div>
        </div>

        {/* Cache Service */}
        <div className="bg-white rounded-lg shadow-sm border p-6">
          <h3 className="text-lg font-medium text-gray-900 mb-4">Cache Service</h3>
          
          <div className="space-y-3">
            <div className="flex items-center justify-between">
              <span className="text-sm text-gray-600">Status:</span>
              <span className={`px-2 py-1 rounded text-xs font-medium ${
                getStatusColor(healthData.services?.cache?.status)
              }`}>
                {healthData.services?.cache?.status}
              </span>
            </div>
            
            {healthData.services?.cache?.stats && (
              <div className="border-t pt-3 space-y-2">
                <div className="flex justify-between text-sm">
                  <span className="text-gray-600">Memory:</span>
                  <span className="font-medium">{healthData.services.cache.stats.used_memory}</span>
                </div>
                <div className="flex justify-between text-sm">
                  <span className="text-gray-600">Keys:</span>
                  <span className="font-medium">{healthData.services.cache.stats.keys}</span>
                </div>
                <div className="flex justify-between text-sm">
                  <span className="text-gray-600">Hit Rate:</span>
                  <span className="font-medium">
                    {healthData.services.cache.stats.hits && healthData.services.cache.stats.misses ? (
                      `${Math.round((healthData.services.cache.stats.hits / (healthData.services.cache.stats.hits + healthData.services.cache.stats.misses)) * 100)}%`
                    ) : 'N/A'}
                  </span>
                </div>
              </div>
            )}
          </div>
        </div>
      </div>

      {/* Feature Flags */}
      <div className="bg-white rounded-lg shadow-sm border p-6">
        <h3 className="text-lg font-medium text-gray-900 mb-4">Feature Status</h3>
        
        <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-6 gap-4">
          {healthData.features && Object.entries(healthData.features).map(([feature, enabled]: [string, any]) => (
            <div key={feature} className="flex items-center space-x-2">
              <span className={`w-2 h-2 rounded-full ${enabled ? 'bg-green-400' : 'bg-gray-300'}`}></span>
              <span className="text-sm text-gray-700 capitalize">
                {feature.replace(/_/g, ' ')}
              </span>
            </div>
          ))}
        </div>
      </div>
    </div>
  )
}