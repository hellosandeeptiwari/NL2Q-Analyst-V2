import { useState } from 'react'
import { useQuery } from '@tanstack/react-query'
import { 
  DatabaseIcon, 
  CpuChipIcon, 
  ChartBarIcon,
  SparklesIcon,
  PlayIcon,
  Cog6ToothIcon
} from '@heroicons/react/24/outline'
import QueryInterface from '../components/QueryInterface'
import StatusPanel from '../components/StatusPanel'
import AnalyticsDashboard from '../components/AnalyticsDashboard'
import { healthService } from '../services/api'

export default function Home() {
  const [activeTab, setActiveTab] = useState('query')
  
  const { data: healthData, isLoading: healthLoading } = useQuery({
    queryKey: ['health'],
    queryFn: healthService.getHealth,
    refetchInterval: 30000, // Refetch every 30 seconds
  })

  const tabs = [
    { id: 'query', name: 'Query', icon: PlayIcon },
    { id: 'analytics', name: 'Analytics', icon: ChartBarIcon },
    { id: 'status', name: 'System Status', icon: Cog6ToothIcon },
  ]

  return (
    <div className="min-h-screen bg-gray-100">
      {/* Header */}
      <header className="bg-white shadow-sm border-b">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between items-center h-16">
            <div className="flex items-center">
              <SparklesIcon className="h-8 w-8 text-indigo-600" />
              <div className="ml-3">
                <h1 className="text-xl font-bold text-gray-900">NL2Q Analyst V2</h1>
                <p className="text-sm text-gray-500">Next-Generation Query Platform</p>
              </div>
            </div>
            
            <div className="flex items-center space-x-4">
              {/* Health Status Indicator */}
              <div className="flex items-center space-x-2">
                <div className={`h-2 w-2 rounded-full ${
                  healthLoading ? 'bg-yellow-400' :
                  healthData?.status === 'healthy' ? 'bg-green-400' :
                  healthData?.status === 'degraded' ? 'bg-yellow-400' : 'bg-red-400'
                }`} />
                <span className="text-sm text-gray-600">
                  {healthLoading ? 'Checking...' : 
                   healthData?.status ? healthData.status.charAt(0).toUpperCase() + healthData.status.slice(1) : 'Unknown'}
                </span>
              </div>
              
              <div className="flex space-x-1 text-sm text-gray-500">
                <DatabaseIcon className="h-4 w-4" />
                <span>{healthData?.services?.database?.available_databases?.length || 0}</span>
              </div>
              
              <div className="flex space-x-1 text-sm text-gray-500">
                <CpuChipIcon className="h-4 w-4" />
                <span>{healthData?.services?.llm?.available_providers?.length || 0}</span>
              </div>
            </div>
          </div>
        </div>
      </header>

      {/* Navigation Tabs */}
      <div className="bg-white border-b">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <nav className="flex space-x-8">
            {tabs.map((tab) => (
              <button
                key={tab.id}
                onClick={() => setActiveTab(tab.id)}
                className={`flex items-center space-x-2 py-4 px-1 border-b-2 text-sm font-medium ${
                  activeTab === tab.id
                    ? 'border-indigo-500 text-indigo-600'
                    : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
                }`}
              >
                <tab.icon className="h-5 w-5" />
                <span>{tab.name}</span>
              </button>
            ))}
          </nav>
        </div>
      </div>

      {/* Main Content */}
      <main className="max-w-7xl mx-auto py-6 px-4 sm:px-6 lg:px-8">
        {activeTab === 'query' && <QueryInterface />}
        {activeTab === 'analytics' && <AnalyticsDashboard />}
        {activeTab === 'status' && <StatusPanel healthData={healthData} />}
      </main>

      {/* Footer */}
      <footer className="bg-white border-t mt-12">
        <div className="max-w-7xl mx-auto py-6 px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between items-center">
            <div className="text-sm text-gray-500">
              NL2Q Analyst V2.0.0 - Next-Generation Natural Language to Query Platform
            </div>
            <div className="flex space-x-6 text-sm text-gray-500">
              <a href="#" className="hover:text-gray-900">Documentation</a>
              <a href="#" className="hover:text-gray-900">API Reference</a>
              <a href="#" className="hover:text-gray-900">GitHub</a>
            </div>
          </div>
        </div>
      </footer>
    </div>
  )
}