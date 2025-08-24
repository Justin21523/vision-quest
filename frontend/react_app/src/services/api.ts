// frontend/react_app/src/services/api.ts
const API_BASE = "http://localhost:8000/api/v1";

export interface CaptionParams {
  max_length: number;
  num_beams: number;
  temperature: number;
}

export interface VQAParams {
  question: string;
  lang: string;
  max_length: number;
}

export const generateCaption = async (file: File, params: CaptionParams) => {
  const formData = new FormData();
  formData.append("file", file);
  formData.append("max_length", params.max_length.toString());
  formData.append("num_beams", params.num_beams.toString());
  formData.append("temperature", params.temperature.toString());

  const response = await fetch(`${API_BASE}/caption/`, {
    method: "POST",
    body: formData,
  });

  if (!response.ok) {
    const error = await response
      .json()
      .catch(() => ({ detail: "Network error" }));
    throw new Error(error.detail || "Caption generation failed");
  }

  return response.json();
};

export const askVQA = async (file: File, params: VQAParams) => {
  const formData = new FormData();
  formData.append("file", file);
  formData.append("question", params.question);
  formData.append("lang", params.lang);
  formData.append("max_length", params.max_length.toString());

  const response = await fetch(`${API_BASE}/vqa/`, {
    method: "POST",
    body: formData,
  });

  if (!response.ok) {
    const error = await response
      .json()
      .catch(() => ({ detail: "Network error" }));
    throw new Error(error.detail || "VQA failed");
  }

  return response.json();
};

export const getHealthStatus = async () => {
  const response = await fetch(`${API_BASE}/health/`);

  if (!response.ok) {
    throw new Error("Health check failed");
  }

  return response.json();
};
