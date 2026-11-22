/** Типы ответов API «Ревизор». Отражают backend/main.py один в один. */

export type Level = 'confirmed' | 'likely_real' | 'fog' | 'doubtful' | 'likely_fake';

export type Plan = 'free' | 'pro' | 'team';

export type Verdict = 'правда' | 'фейк' | 'непонятно';

export interface Signal {
  id: string;
  name: string;
  stage: string;
  verdict: 'supports_real' | 'supports_fake' | 'neutral' | 'unavailable';
  strength: number;
  reliability: number;
  summary: string;
  details: Record<string, unknown>;
}

export interface DebateThesis {
  text: string;
  signal_ids: string[];
}

export interface ProvenanceEntry {
  date: string | null;
  where: string;
  note: string;
}

export interface NextStep {
  kind: string;
  title: string;
  why: string;
}

/** Ответ проверки. В free-тарифе разборных полей нет — это не баг, а тариф. */
export interface CheckResponse {
  id: string;
  verdict: Verdict;
  level: Level;
  level_label: string;
  score: number;
  uncertainty: number;
  plan: Plan;
  upgrade_hint?: string;
  quota_left?: number | null;
  signals?: Signal[];
  debate?: { for_real: DebateThesis[]; for_fake: DebateThesis[] };
  provenance?: ProvenanceEntry[];
  next_steps?: NextStep[];
  limits?: string[];
  duration_ms?: number;
  /** Постоянная ссылка на публичный паспорт утверждения. */
  passport?: string;
  /** Разобранные утверждения — только в pro-тарифе. */
  claims?: ClaimFinding[];
  sources_used?: number;
  /** Объяснение по фото или видео: от модели или от детекторов. */
  explanation?: string;
  explanation_source?: 'model' | 'detectors';
  model_verdict?: 'real' | 'fake' | 'unclear';
  /** Тема утверждения: к какому классу отнесено и почему. */
  tag?: TagInfo;
  /** Рубрика: из чего сложился балл доверия. */
  reasoning?: Reasoning;
}

export interface TagInfo {
  key: string;
  title: string;
  score?: number;
  hits?: string[];
  risk?: number;
  rules?: string[];
}

export interface Reasoning {
  confidence: number;
  verdict: string;
  summary: string;
  tag?: { key: string; title: string };
  rules_applied?: string[];
  criteria: ReasoningCriterion[];
}

export interface ReasoningCriterion {
  key: string;
  title: string;
  points: number;
  max_points: number;
  reason: string;
  tone: 'ok' | 'warn' | 'bad';
}

export interface ClaimFinding {
  text: string;
  status: 'supported' | 'contradicted' | 'context' | 'insufficient';
  explanation: string;
  quote: string;
  source_title: string;
  source_url: string;
  trusted: boolean;
  limitations: string[];
  /** Что делать дальше — с тарифа «Плюс». */
  action?: string;
  /** Готовая фраза для пересылки — только на «Максе». */
  resend?: string;
}

export interface TopItem {
  text_hash: string;
  sample_text: string;
  checks_count: number;
  last_at: string;
  verdict: Verdict;
  score: number;
  level: Level;
  level_label: string;
  verdict_variants: number;
  pinned: boolean;
  /** Сколько проверок за последний час и минуту — «живость» новости. */
  last_hour?: number;
  last_minute?: number;
}

export interface HistoryItem {
  id: string;
  text: string;
  verdict: Verdict;
  level: Level;
  level_label: string;
  score: number;
  uncertainty: number;
  text_hash: string;
  created_at: string;
  passport: string;
}

export interface TopResponse {
  window_hours: number;
  items: TopItem[];
}

/** Лента проверенного: что разобрали и с каким итогом. */
export interface FeedItem {
  text_hash: string;
  sample_text: string;
  checks_count: number;
  last_at: string;
  verdict: Verdict;
  score: number;
  level: Level;
  level_label: string;
  passport: string;
}

export interface FeedResponse {
  window_hours: number;
  items: FeedItem[];
  verified: number;
  refuted: number;
  unclear: number;
}

/** Корпоративный ключ: сам ключ приходит только при выпуске. */
export interface ApiKey {
  id: string;
  key?: string;
  prefix: string;
  label: string;
  created_at: string;
  last_used_at: string | null;
  revoked_at: string | null;
}

export interface ApiKeyUsage {
  days: number;
  checks: number;
}

export interface FeedbackStats {
  days: number;
  total: number;
  ok: number;
  wrong: number;
  accuracy: number | null;
  by_level: Record<string, { total: number; ok: number; wrong: number }>;
}

export interface Stats {
  checks: number;
  unique_claims: number;
  fake: number;
  real: number;
  fog: number;
  pro_users: number;
}

export interface PlanInfo {
  id: Plan;
  name: string;
  price_rub: number;
  price_year_rub?: number;
  limits: string;
  features: string[];
  recommended?: boolean;
}

export interface Me {
  id: string;
  plan: Plan;
  email?: string | null;
  name?: string;
  expires_at?: string | null;
  created_at?: string | null;
  quota_left?: number;
  quota_per_day?: number;
}

export interface Payment {
  id: string;
  plan: string;
  amount_rub: number;
  status: 'pending' | 'paid' | 'failed';
  method: 'card' | 'sbp' | 'invoice';
  created_at: string;
  confirmed_at?: string | null;
  plan_name?: string;
  days?: number;
}

export interface PricingResponse {
  plans: PlanInfo[];
  note: string;
  trial?: { days: number; note: string };
  demo_warning?: string;
}

export interface ApiErrorBody {
  error: { code: string; message: string };
}

/** Автопилот: фоновый контур сбора и проверки. */
export interface AutopilotLabelCounts {
  fake?: number;
  real?: number;
  uncertain?: number;
}

export interface AutopilotModel {
  version?: string;
  promoted_at?: string;
  created_at?: number;
  f1?: number;
  precision_score?: number;
  recall?: number;
  n_train?: number;
}

export interface AutopilotSource {
  source_id: string;
  source_name: string;
  role: string;
  n: number;
}

export interface AutopilotStats {
  available: boolean;
  items_total: number;
  by_label: AutopilotLabelCounts;
  fake: number;
  real: number;
  uncertain: number;
  archived: number;
  pending: number;
  by_status?: Record<string, number>;
  model: AutopilotModel | null;
  llm: { last_hour: number; today: number };
  sources: AutopilotSource[];
}

export interface AutopilotItem {
  id: number;
  source_id: string;
  source_name: string;
  url: string;
  title: string;
  excerpt: string;
  label: 'fake' | 'real' | 'uncertain';
  label_confidence: number;
  label_reason: string;
  label_source: string;
  trust_score: number | null;
  trust_level: string | null;
  verified_at: number | null;
}

export interface AutopilotRecentResponse {
  items: AutopilotItem[];
  total: number;
}
