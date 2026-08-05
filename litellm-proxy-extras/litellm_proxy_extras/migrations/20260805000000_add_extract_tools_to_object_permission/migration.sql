-- Add extract tool permissions to object permissions.
ALTER TABLE "LiteLLM_ObjectPermissionTable" ADD COLUMN IF NOT EXISTS "extract_tools" TEXT[] DEFAULT ARRAY[]::TEXT[];
