import { useState } from 'react'
import { 
  DocumentTextIcon,
  TableCellsIcon,
  ChartBarIcon,
  ClockIcon,
  CpuChipIcon,
  DatabaseIcon,
  SparklesIcon
} from '@heroicons/react/24/outline'

interface QueryResultsProps {
  queryResult: any
  onNewQuery: () => void
}

export default function QueryResults({ queryResult, onNewQuery }: QueryResultsProps) {
  const [activeView, setActiveView] = useState('table')

  const { sql, results, execution_time, llm_provider, from_cache } = queryResult

  return (
    <div className="bg-white rounded-lg shadow-sm border overflow-hidden">
      {/* Header */}
      <div className="px-6 py-4 border-b bg-gray-50">
        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-3">
            <SparklesIcon className="h-5 w-5 text-green-600" />
            <h3 className="text-lg font-semibold text-gray-900">Query Results</h3>
            {from_cache && (
              <span className="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium bg-blue-100 text-blue-800">
                Cached Result
              </span>
            )}
          </div>
          
          <button
            onClick={onNewQuery}
            className="text-sm text-indigo-600 hover:text-indigo-900 font-medium"
          >
            New Query
          </button>
        </div>
        
        {/* Metadata */}
        <div className="flex items-center space-x-6 mt-3 text-sm text-gray-600">
          <div className="flex items-center space-x-1">
            <ClockIcon className="h-4 w-4" />
            <span>{execution_time.toFixed(2)}s</span>
          </div>
          
          <div className="flex items-center space-x-1">
            <CpuChipIcon className="h-4 w-4" />
            <span>{llm_provider}</span>
          </div>
          
          <div className="flex items-center space-x-1">
            <TableCellsIcon className="h-4 w-4" />
            <span>{results.row_count} rows</span>
          </div>
          
          <div className="flex items-center space-x-1">
            <DatabaseIcon className="h-4 w-4" />
            <span>{results.database_id || 'default'}</span>
          </div>
        </div>
      </div>

      {/* View Tabs */}
      <div className="border-b">
        <nav className="flex space-x-6 px-6">
          {[
            { id: 'table', name: 'Table View', icon: TableCellsIcon },
            { id: 'sql', name: 'Generated SQL', icon: DocumentTextIcon },
            { id: 'chart', name: 'Visualization', icon: ChartBarIcon },
          ].map((tab) => (
            <button
              key={tab.id}
              onClick={() => setActiveView(tab.id)}
              className={`flex items-center space-x-2 py-3 px-1 border-b-2 text-sm font-medium ${
                activeView === tab.id
                  ? 'border-indigo-500 text-indigo-600'
                  : 'border-transparent text-gray-500 hover:text-gray-700'
              }`}
            >
              <tab.icon className="h-4 w-4" />
              <span>{tab.name}</span>
            </button>
          ))}
        </nav>
      </div>

      {/* Content */}
      <div className="p-6">
        {activeView === 'table' && (
          <TableView results={results} />
        )}
        
        {activeView === 'sql' && (
          <SQLView sql={sql} />
        )}
        
        {activeView === 'chart' && (
          <ChartView results={results} />
        )}
      </div>
    </div>
  )
}

function TableView({ results }: { results: any }) {
  if (!results.rows || results.rows.length === 0) {
    return (
      <div className="text-center py-12 text-gray-500">
        <TableCellsIcon className="h-12 w-12 mx-auto mb-4 text-gray-300" />
        <p>No data to display</p>
      </div>
    )
  }

  return (
    <div className="overflow-x-auto">
      <table className="min-w-full divide-y divide-gray-200">
        <thead className="bg-gray-50">
          <tr>
            {results.columns.map((column: string) => (
              <th 
                key={column}
                className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider"
              >
                {column}
              </th>
            ))}
          </tr>
        </thead>
        <tbody className="bg-white divide-y divide-gray-200">
          {results.rows.slice(0, 50).map((row: any, index: number) => (
            <tr key={index} className="hover:bg-gray-50">
              {results.columns.map((column: string) => (
                <td 
                  key={column}
                  className="px-6 py-4 whitespace-nowrap text-sm text-gray-900"
                >
                  {formatCellValue(row[column])}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
      
      {results.rows.length > 50 && (
        <div className="mt-4 text-center text-sm text-gray-500">
          Showing first 50 of {results.row_count} rows
        </div>
      )}
    </div>
  )
}

function SQLView({ sql }: { sql: string }) {
  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between">
        <h4 className="text-sm font-medium text-gray-900">Generated SQL Query</h4>
        <button 
          onClick={() => navigator.clipboard.writeText(sql)}
          className="text-sm text-indigo-600 hover:text-indigo-900"
        >
          Copy to Clipboard
        </button>
      </div>
      
      <div className="bg-gray-900 rounded-lg p-4 overflow-x-auto">
        <pre className="text-sm text-green-400 font-mono whitespace-pre-wrap">
          {sql}
        </pre>
      </div>
      
      <div className="text-sm text-gray-600">
        <p>This SQL query was automatically generated from your natural language input using AI.</p>
      </div>
    </div>
  )
}

function ChartView({ results }: { results: any }) {
  // Simple chart placeholder - in a real implementation, this would use Plotly or Recharts
  return (
    <div className="text-center py-12">
      <ChartBarIcon className="h-16 w-16 mx-auto mb-4 text-gray-300" />
      <h4 className="text-lg font-medium text-gray-900 mb-2">Visualization</h4>
      <p className="text-gray-600 mb-4">
        Interactive charts and visualizations would appear here based on your query results.
      </p>
      
      {/* Mock Chart Data Summary */}
      {results.rows && results.rows.length > 0 && (
        <div className="mt-6 bg-gray-50 rounded-lg p-4 text-left max-w-md mx-auto">
          <h5 className="font-medium text-gray-900 mb-2">Data Summary</h5>
          <ul className="text-sm text-gray-600 space-y-1">
            <li>• {results.row_count} data points</li>
            <li>• {results.columns.length} dimensions</li>
            <li>• Suitable for: {suggestChartType(results.columns)}</li>
          </ul>
        </div>
      )}
    </div>
  )
}

function formatCellValue(value: any): string {
  if (value === null || value === undefined) {
    return ''
  }
  
  if (typeof value === 'number') {
    return value.toLocaleString()
  }
  
  if (typeof value === 'boolean') {
    return value ? 'Yes' : 'No'
  }
  
  return String(value)
}

function suggestChartType(columns: string[]): string {
  const hasDate = columns.some(col => 
    col.toLowerCase().includes('date') || 
    col.toLowerCase().includes('time')
  )
  
  const hasNumeric = columns.some(col =>
    col.toLowerCase().includes('count') ||
    col.toLowerCase().includes('amount') ||
    col.toLowerCase().includes('price') ||
    col.toLowerCase().includes('revenue')
  )
  
  if (hasDate && hasNumeric) {
    return 'Time series chart'
  } else if (hasNumeric) {
    return 'Bar chart or pie chart'
  } else {
    return 'Table or category chart'
  }
}