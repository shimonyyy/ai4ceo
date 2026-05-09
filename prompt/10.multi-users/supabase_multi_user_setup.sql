-- Supabase 멀티유저(user_id 기반) 스키마 생성 SQL
-- multi-users-ref.py 기준 테이블/함수
-- Supabase SQL Editor에서 그대로 실행하세요.

-- 0) 확장
create extension if not exists vector;
create extension if not exists pgcrypto;

-- 1) 커스텀 로그인 사용자 테이블 (Supabase Auth 미사용)
create table if not exists public.app_users (
    id uuid primary key default gen_random_uuid(),
    login_id text not null unique,
    password_hash text not null,
    created_at timestamptz not null default now()
);

create index if not exists app_users_login_id_idx on public.app_users(login_id);

-- 2) 세션 테이블
create table if not exists public.sessions (
    id uuid primary key,
    user_id uuid not null references public.app_users(id) on delete cascade,
    title text not null default '새 세션',
    model_name text,
    processed_files jsonb not null default '[]'::jsonb,
    created_at timestamptz not null default now(),
    updated_at timestamptz not null default now()
);

create index if not exists sessions_user_id_idx on public.sessions(user_id);
create index if not exists sessions_updated_at_idx on public.sessions(updated_at desc);

-- 3) 메시지 테이블
create table if not exists public.messages (
    id uuid primary key default gen_random_uuid(),
    session_id uuid not null references public.sessions(id) on delete cascade,
    user_id uuid not null references public.app_users(id) on delete cascade,
    role text not null check (role in ('user', 'assistant', 'system')),
    content text not null,
    position integer not null default 0,
    created_at timestamptz not null default now()
);

create index if not exists messages_session_user_idx on public.messages(session_id, user_id);
create index if not exists messages_position_idx on public.messages(session_id, position);

-- 4) 문서 청크(벡터) 테이블
create table if not exists public.document_chunks (
    id uuid primary key default gen_random_uuid(),
    user_id uuid not null references public.app_users(id) on delete cascade,
    content_hash text not null unique,
    file_name text not null,
    content text not null,
    metadata jsonb not null default '{}'::jsonb,
    embedding vector(1536) not null,
    created_at timestamptz not null default now()
);

create index if not exists document_chunks_user_id_idx on public.document_chunks(user_id);
create index if not exists document_chunks_file_name_idx on public.document_chunks(file_name);
create index if not exists document_chunks_embedding_idx
    on public.document_chunks using ivfflat (embedding vector_cosine_ops);

-- 5) 세션-문서 연결 테이블
create table if not exists public.session_documents (
    id uuid primary key default gen_random_uuid(),
    session_id uuid not null references public.sessions(id) on delete cascade,
    user_id uuid not null references public.app_users(id) on delete cascade,
    document_chunk_id uuid not null references public.document_chunks(id) on delete cascade,
    file_name text not null,
    chunk_index integer not null default 0,
    metadata jsonb not null default '{}'::jsonb,
    created_at timestamptz not null default now(),
    unique (session_id, document_chunk_id)
);

create index if not exists session_documents_session_user_idx on public.session_documents(session_id, user_id);
create index if not exists session_documents_file_name_idx on public.session_documents(file_name);

-- 6) updated_at 자동 갱신 트리거
create or replace function public.set_updated_at()
returns trigger
language plpgsql
as $$
begin
    new.updated_at = now();
    return new;
end;
$$;

drop trigger if exists trg_sessions_updated_at on public.sessions;
create trigger trg_sessions_updated_at
before update on public.sessions
for each row
execute function public.set_updated_at();

-- 7) 세션 내 문서 검색 RPC (앱에서 사용: match_session_documents)
create or replace function public.match_session_documents(
    query_embedding vector(1536),
    match_session_id uuid,
    match_user_id uuid,
    match_threshold float default 0.2,
    match_count int default 8
)
returns table (
    document_chunk_id uuid,
    session_id uuid,
    user_id uuid,
    file_name text,
    content text,
    metadata jsonb,
    similarity float
)
language sql
stable
as $$
    select
        dc.id as document_chunk_id,
        sd.session_id,
        sd.user_id,
        sd.file_name,
        dc.content,
        coalesce(sd.metadata, dc.metadata) as metadata,
        1 - (dc.embedding <=> query_embedding) as similarity
    from public.session_documents sd
    join public.document_chunks dc on dc.id = sd.document_chunk_id
    where
        sd.session_id = match_session_id
        and sd.user_id = match_user_id
        and (1 - (dc.embedding <=> query_embedding)) >= match_threshold
    order by dc.embedding <=> query_embedding
    limit match_count;
$$;

-- 8) 호환용 RPC (선택): match_documents
create or replace function public.match_documents(
    query_embedding vector(1536),
    user_id_filter uuid default null,
    session_id_filter uuid default null,
    match_threshold float default 0.2,
    match_count int default 8
)
returns table (
    id uuid,
    user_id uuid,
    session_id uuid,
    file_name text,
    content text,
    metadata jsonb,
    similarity float
)
language sql
stable
as $$
    select
        dc.id,
        sd.user_id,
        sd.session_id,
        sd.file_name,
        dc.content,
        coalesce(sd.metadata, dc.metadata) as metadata,
        1 - (dc.embedding <=> query_embedding) as similarity
    from public.session_documents sd
    join public.document_chunks dc on dc.id = sd.document_chunk_id
    where
        (user_id_filter is null or sd.user_id = user_id_filter)
        and (session_id_filter is null or sd.session_id = session_id_filter)
        and (1 - (dc.embedding <=> query_embedding)) >= match_threshold
    order by dc.embedding <=> query_embedding
    limit match_count;
$$;

-- 참고:
-- - 현재 앱은 service role 키 기반으로 서버측에서 user_id를 강제 필터링합니다.
-- - Supabase Auth를 쓰지 않으므로 auth.uid() 기반 RLS 정책은 이 스키마와 맞지 않습니다.
-- - 필요 시 별도 앱 토큰 체계를 도입한 후 RLS 정책을 추가하세요.
