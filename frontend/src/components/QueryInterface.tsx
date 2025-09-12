import { useState } from 'react'
import { useQuery, useMutation } from '@tanstack/react-query'
import { 
  PaperAirplaneIcon, 
  CpuChipIcon,
  ClockIcon,
  DatabaseIcon,
  SparklesIcon,
  DocumentTextIcon
} from '@heroicons/react/24/outline'
import { queryService, databaseService, llmService } from '../services/api'
import QueryResults from './QueryResults'
import toast from 'react-hot-toast'

export default function QueryInterface() {
  const [naturalLanguageQuery, setNaturalLanguageQuery] = useState('')
  const [selectedDatabase, setSelectedDatabase] = useState('')
  const [selectedProvider, setSelectedProvider] = useState('')
  const [lastQueryResult, setLastQueryResult] = useState(null)

  // Fetch available databases
  const { data: databases } = useQuery({
    queryKey: ['databases'],
    queryFn: databaseService.getConnections,
  })

  // Fetch available LLM providers
  const { data: providers } = useQuery({
    queryKey: ['llm-providers'],
    queryFn: llmService.getProviders,
  })

  // Query execution mutation
  const executeQueryMutation = useMutation({
    mutationFn: queryService.executeQuery,
    onSuccess: (data) => {
      setLastQueryResult(data)
      toast.success(`Query executed successfully! ${data.results.row_count} rows returned`)
    },
    onError: (error: any) => {
      toast.error(`Query failed: ${error.response?.data?.message || error.message}`)
    },
  })

  const handleExecuteQuery = async () => {
    if (!naturalLanguageQuery.trim()) {
      toast.error('Please enter a natural language query')
      return
    }

    executeQueryMutation.mutate({
      natural_language: naturalLanguageQuery,
      database_id: selectedDatabase || undefined,
      llm_provider: selectedProvider || undefined,
      optimization_level: 'standard',
      max_rows: 100,
    })
  }

  const exampleQueries = [
    "Show me the top 10 customers by total purchase amount",
    "Analyze sales trends over the last 6 months", 
    "Find products that haven't sold in the last 90 days",
    "Compare revenue by product category this year vs last year",
    "Identify customers with the highest order frequency"
  ]

  return (
    <div className="space-y-6">
      {/* Query Input Section */}
      <div className="bg-white rounded-lg shadow-sm border p-6">
        <div className="flex items-center space-x-2 mb-4">
          <SparklesIcon className="h-5 w-5 text-indigo-600" />
          <h2 className="text-lg font-semibold text-gray-900">Natural Language Query</h2>
        </div>

        {/* Configuration Options */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-4">
          {/* Database Selection */}
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              <DatabaseIcon className="inline h-4 w-4 mr-1" />
              Database Connection
            </label>
            <select 
              value={selectedDatabase}
              onChange={(e) => setSelectedDatabase(e.target.value)}
              className="w-full px-3 py-2 border border-gray-300 rounded-md focus:ring-indigo-500 focus:border-indigo-500"
            >
              <option value="">Auto-select database</option>
              {databases?.map((db: any) => (
                <option key={db.id} value={db.id}>
                  {db.id} ({db.type}) - {db.table_count} tables
                </option>
              ))}
            </select>
          </div>

          {/* LLM Provider Selection */}
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              <CpuChipIcon className="inline h-4 w-4 mr-1" />
              LLM Provider
            </label>
            <select
              value={selectedProvider}
              onChange={(e) => setSelectedProvider(e.target.value)}
              className="w-full px-3 py-2 border border-gray-300 rounded-md focus:ring-indigo-500 focus:border-indigo-500"
            >
              <option value="">Auto-select provider</option>
              {providers?.map((provider: any) => (
                <option key={provider.name} value={provider.name} disabled={!provider.status}>
                  {provider.name} {!provider.status && '(Unavailable)'}
                </option>
              ))}
            </select>
          </div>
        </div>

        {/* Query Input */}
        <div className="mb-4">
          <textarea
            value={naturalLanguageQuery}
            onChange={(e) => setNaturalLanguageQuery(e.target.value)}
            placeholder="Enter your natural language query here... e.g., 'Show me sales by region for last month'"
            rows={4}
            className="w-full px-3 py-2 border border-gray-300 rounded-md focus:ring-indigo-500 focus:border-indigo-500 resize-none"
          />
        </div>

        {/* Execute Button */}
        <div className="flex justify-between items-center">
          <div className="flex space-x-2">
            {exampleQueries.slice(0, 2).map((query, index) => (
              <button
                key={index}
                onClick={() => setNaturalLanguageQuery(query)}
                className="text-xs px-2 py-1 bg-gray-100 text-gray-600 rounded hover:bg-gray-200 transition-colors"
              >
                Example {index + 1}
              </button>
            ))}
          </div>
          
          <button
            onClick={handleExecuteQuery}
            disabled={executeQueryMutation.isPending || !naturalLanguageQuery.trim()}
            className="inline-flex items-center space-x-2 px-6 py-2 bg-indigo-600 text-white rounded-md hover:bg-indigo-700 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
          >
            {executeQueryMutation.isPending ? (
              <>
                <ClockIcon className="h-4 w-4 animate-spin" />
                <span>Executing...</span>
              </>
            ) : (
              <>
                <PaperAirplaneIcon className="h-4 w-4" />
                <span>Execute Query</span>
              </>
            )}
          </button>
        </div>
      </div>

      {/* Example Queries */}
      <div className="bg-white rounded-lg shadow-sm border p-6">
        <div className="flex items-center space-x-2 mb-3">
          <DocumentTextIcon className="h-5 w-5 text-gray-600" />
          <h3 className="text-md font-medium text-gray-900">Example Queries</h3>
        </div>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-2">
          {exampleQueries.map((query, index) => (
            <button
              key={index}
              onClick={() => setNaturalLanguageQuery(query)}
              className="text-left p-3 text-sm bg-gray-50 rounded border hover:bg-gray-100 transition-colors"
            >
              "{query}"
            </button>
          ))}
        </div>
      </div>

      {/* Query Results */}
      {lastQueryResult && (
        <QueryResults 
          queryResult={lastQueryResult}
          onNewQuery={() => setLastQueryResult(null)}
        />
      )}

      {/* Loading State */}
      {executeQueryMutation.isPending && (
        <div className="bg-white rounded-lg shadow-sm border p-6">
          <div className="flex items-center justify-center space-x-3">
            <ClockIcon className="h-5 w-5 text-indigo-600 animate-spin" />
            <span className="text-gray-600">Processing your natural language query...</span>
          </div>
          <div className="mt-3 text-center text-sm text-gray-500">
            Converting to SQL and executing against your database
          </div>
        </div>
      )}
    </div>
  )
}