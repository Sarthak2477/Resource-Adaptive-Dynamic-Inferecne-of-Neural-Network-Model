import axios from 'axios';
import type { Experiment, ExperimentDetail, LiveResources, ModelInfo, RunJob, SimulationResult, SystemInfo } from './types';
const http = axios.create({ baseURL: '/api', timeout: 30_000 });
export const api = {
  system: async () => (await http.get<SystemInfo>('/system/info')).data,
  resources: async () => (await http.get<LiveResources>('/system/resources')).data,
  model: async () => (await http.get<ModelInfo>('/model/configurations')).data,
  experiments: async () => (await http.get<Experiment[]>('/experiments')).data,
  experiment: async (id: string) => (await http.get<ExperimentDetail>(`/experiments/${encodeURIComponent(id)}`)).data,
  simulate: async (body: { scenario: string; num_samples: number; deadline_ms: number }) => (await http.post<SimulationResult>('/simulations', body)).data,
  start: async (body: { num_samples: number; deadline_ms: number; device: string; trace: string; latency_policy: string; seed: number; threads: number }) => (await http.post<RunJob>('/experiments', body)).data,
  stop: async (id: string) => (await http.post<RunJob>(`/runs/${id}/stop`)).data,
};
export const exportUrl = (id: string, format: 'json' | 'csv') => `/api/experiments/${encodeURIComponent(id)}/export?format=${format}`;
