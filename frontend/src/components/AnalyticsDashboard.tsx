import { useQuery } from '@tanstack/react-query'
import { 
  ChartBarIcon,
  UsersIcon,
  DatabaseIcon,
  CpuChipIcon,
  TrendingUpIcon,
  ExclamationTriangleIcon
} from '@heroicons/react/24/outline'
import { analyticsService } from '../services/api'

export default function AnalyticsDashboard() {
  const { data: dashboardData, isLoading } = useQuery({
    queryKey: ['analytics-dashboard'],
    queryFn: analyticsService.getDashboard,
    refetchInterval: 60000, // Refresh every minute
  })

  if (isLoading) {
    return <div className="animate-pulse space-y-6">
      {[1, 2, 3].map(i => (
        <div key={i} className="bg-white rounded-lg shadow-sm border p-6">
          <div className="h-4 bg-gray-200 rounded w-1/4 mb-4"></div>
          <div className="space-y-2">
            <div className="h-3 bg-gray-200 rounded"></div>
            <div className="h-3 bg-gray-200 rounded w-5/6"></div>
          </div>
        </div>
      ))}
    </div>
  }

  return (
    <div className="space-y-6">
      {/* Overview Cards */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-6">
        <MetricCard
          title="Total Queries"
          value={dashboardData?.query_analytics?.total_queries?.toLocaleString()}
          icon={ChartBarIcon}
          color="blue"
        />
        
        <MetricCard
          title="Success Rate"
          value={`${(dashboardData?.query_analytics?.success_rate * 100)?.toFixed(1)}%`}
          icon={TrendingUpIcon}
          color="green"
        />
        
        <MetricCard
          title="Active Users"
          value={dashboardData?.user_insights?.active_users}
          icon={UsersIcon}
          color="purple"
        />
        
        <MetricCard
          title="Avg Response"
          value={`${dashboardData?.query_analytics?.avg_execution_time}s`}
          icon={CpuChipIcon}
          color="orange"
        />
      </div>

      {/* Query Analytics */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <div className="bg-white rounded-lg shadow-sm border p-6">
          <h3 className="text-lg font-medium text-gray-900 mb-4">Query Types Distribution</h3>
          
          {dashboardData?.query_analytics?.top_query_types?.map((type: any) => (
            <div key={type.type} className="mb-3">
              <div className="flex justify-between text-sm mb-1">
                <span className="capitalize text-gray-700">{type.type}</span>
                <span className="text-gray-500">{type.percentage}%</span>
              </div>
              <div className="w-full bg-gray-200 rounded-full h-2">
                <div 
                  className="bg-indigo-600 h-2 rounded-full"
                  style={{ width: `${type.percentage}%` }}
                ></div>
              </div>
              <div className="text-xs text-gray-500 mt-1">
                {type.count?.toLocaleString()} queries
              </div>
            </div>
          ))}
        </div>

        <div className="bg-white rounded-lg shadow-sm border p-6">
          <h3 className="text-lg font-medium text-gray-900 mb-4">Most Queried Tables</h3>
          
          <div className="space-y-3">
            {dashboardData?.data_insights?.most_queried_tables?.map((table: any, index: number) => (
              <div key={table.table} className="flex items-center justify-between">
                <div className="flex items-center space-x-3">
                  <span className="flex-shrink-0 w-6 h-6 bg-gray-100 rounded-full flex items-center justify-center text-xs font-medium">
                    {index + 1}
                  </span>
                  <div>
                    <div className="text-sm font-medium text-gray-900">{table.table}</div>
                    <div className="text-xs text-gray-500">{table.query_count} queries</div>
                  </div>
                </div>
                <DatabaseIcon className="h-4 w-4 text-gray-400" />
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* User Insights */}
      <div className="bg-white rounded-lg shadow-sm border p-6">
        <h3 className="text-lg font-medium text-gray-900 mb-4">User Activity Insights</h3>
        
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
          <div>
            <h4 className="text-sm font-medium text-gray-700 mb-2">Peak Activity Hours</h4>
            <div className="flex space-x-2">
              {dashboardData?.user_insights?.most_active_hours?.map((hour: string) => (
                <span key={hour} className="px-2 py-1 bg-blue-100 text-blue-800 text-xs rounded">
                  {hour}
                </span>
              ))}
            </div>
          </div>
          
          <div>
            <h4 className="text-sm font-medium text-gray-700 mb-2">Avg Queries per User</h4>
            <div className="text-2xl font-bold text-gray-900">
              {dashboardData?.user_insights?.queries_per_user?.toFixed(1)}
            </div>
          </div>
          
          <div>
            <h4 className="text-sm font-medium text-gray-700 mb-2">Usage Patterns</h4>
            <div className="text-xs text-gray-600 space-y-1">
              {dashboardData?.user_insights?.common_patterns?.slice(0, 2).map((pattern: string, index: number) => (
                <div key={index}>• {pattern}</div>
              ))}
            </div>
          </div>
        </div>
      </div>

      {/* Data Quality Alerts */}
      {dashboardData?.data_insights?.data_quality_alerts && (
        <div className="bg-white rounded-lg shadow-sm border p-6">
          <div className="flex items-center space-x-2 mb-4">
            <ExclamationTriangleIcon className="h-5 w-5 text-amber-500" />
            <h3 className="text-lg font-medium text-gray-900">Data Quality Alerts</h3>
          </div>
          
          <div className="space-y-3">
            {dashboardData.data_insights.data_quality_alerts.map((alert: any, index: number) => (
              <div key={index} className={`p-3 rounded-lg border-l-4 ${
                alert.severity === 'high' ? 'border-red-400 bg-red-50' :
                alert.severity === 'medium' ? 'border-yellow-400 bg-yellow-50' :
                'border-blue-400 bg-blue-50'
              }`}>
                <div className="flex justify-between items-start">
                  <div>
                    <div className="font-medium text-gray-900">{alert.table}</div>
                    <div className="text-sm text-gray-700">{alert.issue}</div>
                  </div>
                  <span className={`px-2 py-1 rounded text-xs font-medium ${
                    alert.severity === 'high' ? 'bg-red-100 text-red-800' :
                    alert.severity === 'medium' ? 'bg-yellow-100 text-yellow-800' :
                    'bg-blue-100 text-blue-800'
                  }`}>
                    {alert.severity}
                  </span>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  )
}

interface MetricCardProps {
  title: string
  value: string | number | undefined
  icon: any
  color: 'blue' | 'green' | 'purple' | 'orange'
}

function MetricCard({ title, value, icon: Icon, color }: MetricCardProps) {
  const colorClasses = {
    blue: 'text-blue-600 bg-blue-100',
    green: 'text-green-600 bg-green-100',
    purple: 'text-purple-600 bg-purple-100',
    orange: 'text-orange-600 bg-orange-100',
  }

  return (
    <div className="bg-white rounded-lg shadow-sm border p-6">
      <div className="flex items-center">
        <div className={`p-2 rounded-lg ${colorClasses[color]}`}>
          <Icon className="h-6 w-6" />
        </div>
        <div className="ml-4">
          <div className="text-2xl font-bold text-gray-900">
            {value || 'N/A'}
          </div>
          <div className="text-sm text-gray-600">{title}</div>
        </div>
      </div>
    </div>
  )
}