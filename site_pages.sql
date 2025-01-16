-- Enable the pgvector extension
create extension if not exists vector;

-- Drop existing functions if they exist
drop function if exists match_site_pages;
drop function if exists match_nita_pages;

-- Create the documentation chunks table
create table if not exists site_pages (
    id bigserial primary key,
    url varchar not null,
    chunk_number integer not null,
    title varchar not null,
    summary varchar not null,
    content text not null,
    metadata jsonb not null default '{}'::jsonb,
    embedding vector(1024),  -- Mistral embeddings are 1024 dimensions
    created_at timestamp with time zone default timezone('utc'::text, now()) not null,
    
    -- Add a unique constraint to prevent duplicate chunks for the same URL
    unique(url, chunk_number)
);

-- Create an index for better vector similarity search performance
drop index if exists site_pages_embedding_idx;
create index site_pages_embedding_idx on site_pages using ivfflat (embedding vector_cosine_ops);

-- Create an index on metadata for faster filtering
drop index if exists idx_site_pages_metadata;
create index idx_site_pages_metadata on site_pages using gin (metadata);

-- Create a function to search for documentation chunks
create function match_site_pages (
  query_embedding vector(1024),
  match_count int default 10,
  filter jsonb DEFAULT '{}'::jsonb
) returns table (
  id bigint,
  url varchar,
  chunk_number integer,
  title varchar,
  summary varchar,
  content text,
  metadata jsonb,
  similarity float
)
language plpgsql
as $$
#variable_conflict use_column
begin
  return query
  select
    id,
    url,
    chunk_number,
    title,
    summary,
    content,
    metadata,
    1 - (site_pages.embedding <=> query_embedding) as similarity
  from site_pages
  where metadata @> filter
  order by site_pages.embedding <=> query_embedding
  limit match_count;
end;
$$;

-- Enable RLS on the table
alter table site_pages enable row level security;

-- Drop existing policies
drop policy if exists "Allow public read access" on site_pages;

-- Create a policy that allows anyone to read
create policy "Allow public read access"
  on site_pages
  for select
  to public
  using (true);

-- Create the NITA documentation chunks table
create table if not exists nita_pages (
    id bigserial primary key,
    url varchar not null,
    chunk_number integer not null,
    title varchar not null,
    summary varchar not null,
    content text not null,
    metadata jsonb not null default '{}'::jsonb,
    embedding vector(1024),  -- Mistral embeddings are 1024 dimensions
    created_at timestamp with time zone default timezone('utc'::text, now()) not null,
    
    -- Add a unique constraint to prevent duplicate chunks for the same URL
    unique(url, chunk_number)
);

-- Create an index for better vector similarity search performance
drop index if exists nita_pages_embedding_idx;
create index nita_pages_embedding_idx on nita_pages using ivfflat (embedding vector_cosine_ops);

-- Create an index on metadata for faster filtering
drop index if exists idx_nita_pages_metadata;
create index idx_nita_pages_metadata on nita_pages using gin (metadata);

-- Create a function to search for documentation chunks
create function match_nita_pages (
  query_embedding vector(1024),
  match_count int default 10,
  filter jsonb DEFAULT '{}'::jsonb
) returns table (
  id bigint,
  url varchar,
  chunk_number integer,
  title varchar,
  summary varchar,
  content text,
  metadata jsonb,
  similarity float
)
language plpgsql
as $$
#variable_conflict use_column
begin
  return query
  select
    id,
    url,
    chunk_number,
    title,
    summary,
    content,
    metadata,
    1 - (nita_pages.embedding <=> query_embedding) as similarity
  from nita_pages
  where metadata @> filter
  order by nita_pages.embedding <=> query_embedding
  limit match_count;
end;
$$;

-- Enable RLS on the table
alter table nita_pages enable row level security;

-- Drop existing policies
drop policy if exists "Allow public read access" on nita_pages;

-- Create a policy that allows anyone to read
create policy "Allow public read access"
  on nita_pages
  for select
  to public
  using (true);