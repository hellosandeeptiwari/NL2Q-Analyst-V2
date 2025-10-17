import React from 'react';
import { FiLoader, FiClock, FiBarChart } from 'react-icons/fi';
import './ProgressIndicator.css';

interface ProgressStep {
  id: string;
  name: string;
  status: 'pending' | 'running' | 'completed' | 'error';
  startTime?: number;
  endTime?: number;
  progress?: number;
  description?: string;  // Added for showing task details (e.g., email recipients)
}

interface ProgressIndicatorProps {
  steps: ProgressStep[];
  currentStep?: string;
  totalEstimatedTime?: number;
  elapsedTime?: number;
  showTimeEstimate?: boolean;
}

const ProgressIndicator: React.FC<ProgressIndicatorProps> = ({
  steps,
  currentStep,
  totalEstimatedTime,
  elapsedTime = 0,
  showTimeEstimate = true
}) => {
  const completedSteps = steps.filter(step => step.status === 'completed').length;
  const totalSteps = steps.length;
  const overallProgress = totalSteps > 0 ? (completedSteps / totalSteps) * 100 : 0;
  const isCompleted = completedSteps === totalSteps && totalSteps > 0;
  
  const formatTime = (seconds: number): string => {
    if (seconds < 60) return `${Math.round(seconds)}s`;
    const minutes = Math.floor(seconds / 60);
    const remainingSeconds = Math.round(seconds % 60);
    return `${minutes}m ${remainingSeconds}s`;
  };

  const estimatedRemaining = !isCompleted && totalEstimatedTime && elapsedTime < totalEstimatedTime 
    ? totalEstimatedTime - elapsedTime 
    : null;

  return (
    <div className="progress-indicator">
      <div className="progress-header">
        <div className="progress-title">
          <FiBarChart className="progress-icon" />
          <span>{isCompleted ? 'Analysis Complete' : 'Analyzing Data'}</span>
        </div>
        
        {showTimeEstimate && (
          <div className="progress-time">
            <FiClock className="time-icon" />
            {isCompleted ? (
              <span className="completed-time">
                Completed in {formatTime(elapsedTime)}
              </span>
            ) : (
              <>
                <span className="elapsed-time">
                  {formatTime(elapsedTime)}
                </span>
                {estimatedRemaining && (
                  <span className="estimated-remaining">
                    / ~{formatTime(totalEstimatedTime!)} remaining
                  </span>
                )}
              </>
            )}
          </div>
        )}
      </div>

      {/* Overall Progress Bar */}
      <div className="overall-progress">
        <div className="progress-bar">
          <div 
            className="progress-fill" 
            style={{ width: `${overallProgress}%` }}
          />
        </div>
        <div className="progress-text">
          {completedSteps} of {totalSteps} steps completed ({Math.round(overallProgress)}%)
        </div>
      </div>

      {/* Individual Steps */}
      <div className="progress-steps">
        {steps.map((step, index) => {
          const isActive = step.id === currentStep;
          const stepProgress = step.progress || 0;
          
          return (
            <div 
              key={step.id} 
              className={`progress-step ${step.status} ${isActive ? 'active' : ''}`}
            >
              <div className="step-indicator">
                <div className="step-number">
                  {step.status === 'completed' ? (
                    <span className="check-mark">✓</span>
                  ) : step.status === 'running' ? (
                    <FiLoader className="spinning" />
                  ) : step.status === 'error' ? (
                    <span className="error-mark">✗</span>
                  ) : (
                    <span>{index + 1}</span>
                  )}
                </div>
                <div className="step-connector" />
              </div>
              
              <div className="step-content">
                <div className="step-name">{step.name}</div>
                
                {step.description && step.status === 'completed' && (
                  <div className="step-description">{step.description}</div>
                )}
                
                {step.status === 'running' && stepProgress > 0 && (
                  <div className="step-progress">
                    <div className="step-progress-bar">
                      <div 
                        className="step-progress-fill" 
                        style={{ width: `${stepProgress}%` }}
                      />
                    </div>
                    <span className="step-progress-text">{Math.round(stepProgress)}%</span>
                  </div>
                )}
                
                {(step.startTime && step.endTime) && (
                  <div className="step-duration">
                    {formatTime((step.endTime - step.startTime) / 1000)}
                  </div>
                )}
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
};

export default ProgressIndicator;
