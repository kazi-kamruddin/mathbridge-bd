create table if not exists public.mathbridge_sessions (
  id uuid primary key,
  owner_hash text not null,
  mode text not null,
  title text not null,
  payload jsonb not null,
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now()
);

create index if not exists mathbridge_sessions_owner_hash_idx
on public.mathbridge_sessions(owner_hash);
