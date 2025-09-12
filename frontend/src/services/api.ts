import axios, { AxiosResponse } from 'axios'

const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000'

// Create axios instance with default config
const api = axios.create({
  baseURL: `${API_BASE_URL}/api/v2`,
  timeout: 30000,
  headers: {
    'Content-Type': 'application/json',
  },
})

// Response interceptor for error handling
api.interceptors.response.use(
  (response) => response,
  (error) => {
    console.error('API Error:', error.response?.data || error.message)
    return Promise.reject(error)
  }
)

// Health Service
export const healthService = {
  getHealth: async () => {
    const response: AxiosResponse = await api.get('/health')
    return response.data
  },
  
  getLiveness: async () => {
    const response: AxiosResponse = await api.get('/health/live')
    return response.data
  },
  
  getReadiness: async () => {
    const response: AxiosResponse = await api.get('/health/ready')
    return response.data
  },
  
  getMetrics: async () => {
    const response: AxiosResponse = await api.get('/health/metrics')
    return response.data
  },
}

// Query Service
export const queryService = {
  executeQuery: async (queryRequest: {
    natural_language: string
    database_id?: string
    llm_provider?: string
    optimization_level?: string
    max_rows?: number
  }) => {
    const response: AxiosResponse = await api.post('/query/execute', queryRequest)
    return response.data
  },
  
  getQueryStatus: async (queryId: string) => {
    const response: AxiosResponse = await api.get(`/query/${queryId}`)
    return response.data
  },
  
  optimizeQuery: async (queryId: string, optimizationParams: any) => {
    const response: AxiosResponse = await api.post(`/query/${queryId}/optimize`, optimizationParams)
    return response.data
  },
  
  explainQuery: async (queryId: string) => {
    const response: AxiosResponse = await api.get(`/query/${queryId}/explain`)
    return response.data
  },
}

// LLM Service
export const llmService = {
  getProviders: async () => {
    const response: AxiosResponse = await api.get('/llm/providers')
    return response.data
  },
  
  selectOptimalLLM: async (request: {
    query_type?: string
    complexity?: string
  }) => {
    const response: AxiosResponse = await api.post('/llm/select', request)
    return response.data
  },
  
  getUsage: async () => {
    const response: AxiosResponse = await api.get('/llm/usage')
    return response.data
  },
  
  runBenchmark: async () => {
    const response: AxiosResponse = await api.get('/llm/benchmark')
    return response.data
  },
}

// Analytics Service
export const analyticsService = {
  generateInsights: async (request: {
    data_summary: any
    context: string
    analysis_type?: string
  }) => {
    const response: AxiosResponse = await api.post('/analytics/insights', request)
    return response.data
  },
  
  getSuggestions: async (request: {
    current_context?: string
    user_history?: string[]
    dataset_context?: string
  }) => {
    const response: AxiosResponse = await api.post('/analytics/suggestions', request)
    return response.data
  },
  
  predictiveAnalytics: async (data: {
    type?: string
    horizon?: string
  }) => {
    const response: AxiosResponse = await api.post('/analytics/predict', data)
    return response.data
  },
  
  getDashboard: async () => {
    const response: AxiosResponse = await api.get('/analytics/dashboard')
    return response.data
  },
}

// Database Service
export const databaseService = {
  getConnections: async () => {
    const response: AxiosResponse = await api.get('/database/connections')
    return response.data
  },
  
  getSchema: async (databaseId: string) => {
    const response: AxiosResponse = await api.get(`/database/${databaseId}/schema`)
    return response.data
  },
  
  getTables: async (databaseId: string) => {
    const response: AxiosResponse = await api.get(`/database/${databaseId}/tables`)
    return response.data
  },
  
  getTableDetails: async (databaseId: string, tableName: string) => {
    const response: AxiosResponse = await api.get(`/database/${databaseId}/tables/${tableName}`)
    return response.data
  },
  
  testConnection: async (databaseId: string) => {
    const response: AxiosResponse = await api.post(`/database/${databaseId}/test`)
    return response.data
  },
  
  clearCache: async (databaseId: string, tenantId: string = 'default') => {
    const response: AxiosResponse = await api.delete(`/database/${databaseId}/cache`, {
      params: { tenant_id: tenantId }
    })
    return response.data
  },
  
  getStatistics: async (databaseId: string) => {
    const response: AxiosResponse = await api.get(`/database/${databaseId}/stats`)
    return response.data
  },
}

export default api