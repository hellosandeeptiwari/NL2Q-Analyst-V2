import React, { useState, useEffect } from 'react';
import axios from 'axios';

interface UserDashboardProps {
  totalQueries?: number;
  insight?: string;
}

function UserDashboard({ totalQueries = 0, insight = '' }: UserDashboardProps) {
  const [favorites, setFavorites] = useState<string[]>([]);

  const addToFavorites = async (query: string) => {
    try {
      setFavorites([...favorites, query]);
    } catch (error) {
      console.error('Failed to add favorite');
    }
  };

  return (
    <div>
      <h2 style={{ color: '#495057', marginBottom: '1.5rem', fontSize: '1.5rem' }}>User Dashboard</h2>

      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(250px, 1fr))', gap: '1rem', marginBottom: '2rem' }}>
        <div style={{ background: '#e7f3ff', padding: '1.5rem', borderRadius: '8px', border: '1px solid #b8daff' }}>
          <h3 style={{ margin: '0 0 0.5rem 0', color: '#495057' }}>Total Queries</h3>
          <div style={{ fontSize: '2rem', fontWeight: 'bold', color: '#007bff' }}>{totalQueries}</div>
        </div>
        <div style={{ background: '#d4edda', padding: '1.5rem', borderRadius: '8px', border: '1px solid #c3e6cb' }}>
          <h3 style={{ margin: '0 0 0.5rem 0', color: '#495057' }}>Active Sessions</h3>
          <div style={{ fontSize: '2rem', fontWeight: 'bold', color: '#28a745' }}>1</div>
        </div>
        <div style={{ background: '#fff3cd', padding: '1.5rem', borderRadius: '8px', border: '1px solid #ffeaa7' }}>
          <h3 style={{ margin: '0 0 0.5rem 0', color: '#495057' }}>Favorites</h3>
          <div style={{ fontSize: '2rem', fontWeight: 'bold', color: '#ffc107' }}>{favorites.length}</div>
        </div>
      </div>

      {insight && (
        <div style={{ background: '#f8f9fa', padding: '1rem', borderRadius: '8px', marginBottom: '1rem', border: '1px solid #dee2e6' }}>
          <h4 style={{ color: '#495057', marginBottom: '0.5rem' }}>Latest Insight</h4>
          <p style={{ margin: 0, color: '#6c757d', fontSize: '0.9rem' }}>{insight}</p>
        </div>
      )}

      <div>
        <h3 style={{ color: '#495057', marginBottom: '1rem' }}>Favorite Queries</h3>
        {favorites.length === 0 ? (
          <div style={{ textAlign: 'center', color: '#6c757d', padding: '2rem' }}>No favorites yet</div>
        ) : (
          <div style={{ display: 'flex', flexWrap: 'wrap', gap: '0.5rem' }}>
            {favorites.map((query, index) => (
              <div key={index} style={{
                background: '#f8f9fa',
                padding: '0.75rem',
                borderRadius: '20px',
                border: '1px solid #dee2e6',
                fontSize: '0.9rem',
                color: '#495057'
              }}>
                {query}
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}

export default UserDashboard;
export {};
