// frontend/react_app/src/components/TaskCard.tsx
import React from 'react'

interface TaskCardProps {
  title: string
  description: string
  children: React.ReactNode
  className?: string
}

export const TaskCard: React.FC<TaskCardProps> = ({
  title,
  description,
  children,
  className = ''
}) => {
  return (
    <div className={`task-card ${className}`}>
      <div className="card-header">
        <h2 className="card-title">{title}</h2>
        <p className="card-description">{description}</p>
      </div>
      <div className="card-content">
        {children}
      </div>
    </div>
  )
}