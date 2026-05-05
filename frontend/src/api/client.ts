import axios from 'axios';
import { InferenceStatus, InferenceResult } from '../types';

export const apiClient = axios.create({ baseURL: '/api/v1' });

export const submitInference = (formData: FormData) => 
    apiClient.post<{ task_id: string }>('/inference', formData);

export const getStatus = (taskId: string) =>
    apiClient.get<InferenceStatus>(`/inference/${taskId}/status`);

export const getResult = (taskId: string) =>
    apiClient.get<InferenceResult>(`/inference/${taskId}/result`);
