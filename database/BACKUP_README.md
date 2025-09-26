# Supabase Database Backup & Restore

这个目录包含了完整的 Supabase 数据库备份和恢复解决方案。

## 文件说明

- `backup_supabase.py` - 主备份脚本，创建完整的数据库备份
- `restore_supabase.py` - 恢复脚本，从备份文件恢复数据库
- `backup/` - 备份文件存储目录
- `BACKUP_README.md` - 本说明文件

## 备份类型

### 1. 完整数据库备份 (pg_dump)
- **文件格式**: `.dump` (二进制) 和 `.sql` (文本)
- **用途**: 完整的数据库恢复，包括结构、数据、索引、序列等
- **恢复方式**: 使用 `pg_restore` 或 `psql`

### 2. JSON 数据备份
- **文件格式**: `.json`
- **用途**: 数据检查、分析、部分表恢复
- **特点**: 人类可读，便于检查数据内容

### 3. 元数据文件
- **文件格式**: `backup_metadata_*.json`
- **内容**: 备份信息、文件大小、恢复说明等

## 环境变量设置

在运行备份脚本前，需要设置以下环境变量：

```bash
# 必需 - Supabase API 访问
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_KEY=your-supabase-anon-or-service-key

# 推荐 - 完整数据库备份 (pg_dump)
DATABASE_URL=postgresql://postgres:password@db.your-project.supabase.co:5432/postgres
```

### 获取 DATABASE_URL

1. 登录 Supabase Dashboard
2. 进入项目设置 → Database
3. 复制 "Connection string" 中的 URI
4. 将 `[YOUR-PASSWORD]` 替换为实际密码

## 使用方法

### 手动备份

```bash
# 完整备份 (推荐)
python database/backup_supabase.py

# 仅 JSON 数据备份
SUPABASE_KEY=your-key python database/backup_supabase.py
```

### 自动备份 (GitHub Actions)

备份会通过 GitHub Actions 自动运行：
- **频率**: 每周一次（每周日凌晨2点 UTC）
- **触发**: 定时任务或手动触发
- **存储**: GitHub Actions Artifacts（备份文件保留90天，日志保留30天）
- **注意**: 仅执行备份操作，不包含自动恢复

### 查看可用备份

```bash
# 列出所有备份文件
python database/restore_supabase.py --list

# 指定备份目录
python database/restore_supabase.py --list --backup-dir /path/to/backups
```

### 恢复数据库 (仅手动操作)

⚠️ **重要提醒**: 数据库恢复操作仅支持手动执行，不会自动运行。请在恢复前仔细确认目标数据库。

```bash
# 从 pg_dump 备份恢复 (推荐)
python database/restore_supabase.py \
  --restore database/backup/supabase_complete_backup_20231227_143022.dump \
  --target-db "postgresql://user:pass@host:5432/target_db"

# 从 SQL 备份恢复
python database/restore_supabase.py \
  --restore database/backup/supabase_backup_20231227_143022.sql \
  --target-db "postgresql://user:pass@host:5432/target_db"
```

**恢复前必须确认**:
1. 目标数据库连接信息正确
2. 有足够的权限执行恢复操作
3. 已在测试环境中验证备份文件
4. 了解恢复操作会覆盖现有数据

## 备份文件命名规则

```
supabase_complete_backup_YYYYMMDD_HHMMSS.dump  # pg_dump 二进制格式
supabase_backup_YYYYMMDD_HHMMSS.sql            # pg_dump SQL 格式
supabase_json_backup_YYYYMMDD_HHMMSS.json      # JSON 数据备份
backup_metadata_YYYYMMDD_HHMMSS.json           # 备份元数据
```

## 恢复选项

### 1. 完整恢复 (推荐)

使用 pg_dump 备份文件可以完整恢复数据库：

```bash
# 使用 pg_restore (二进制格式)
pg_restore --clean --if-exists --no-owner --no-privileges \
  --dbname "postgresql://user:pass@host:5432/target_db" \
  backup_file.dump

# 使用 psql (SQL 格式)
psql "postgresql://user:pass@host:5432/target_db" < backup_file.sql
```

### 2. 选择性恢复

```bash
# 仅恢复特定表
pg_restore --table=properties --table=property_history \
  --dbname "target_db_url" backup_file.dump

# 仅恢复数据 (不包括结构)
pg_restore --data-only --dbname "target_db_url" backup_file.dump
```

### 3. 结构恢复

```bash
# 仅恢复表结构 (不包括数据)
pg_restore --schema-only --dbname "target_db_url" backup_file.dump
```

## 故障排除

### 常见问题

1. **权限错误**
   - 确保使用 `service_role` key 而不是 `anon` key
   - 检查数据库连接权限

2. **网络连接问题**
   - 确认 DATABASE_URL 可访问
   - 检查防火墙设置

3. **磁盘空间不足**
   - 备份文件可能很大，确保有足够空间
   - 考虑使用压缩选项

4. **超时问题**
   - 大型数据库备份可能需要很长时间
   - 调整 GitHub Actions 超时设置

### 日志文件

备份过程会生成日志文件：
- `supabase_backup.log` - 详细的备份日志
- GitHub Actions 日志 - 在 Actions 页面查看

## 安全注意事项

1. **凭据保护**
   - 不要在代码中硬编码数据库密码
   - 使用 GitHub Secrets 存储敏感信息

2. **备份加密**
   - 考虑对备份文件进行加密
   - 使用安全的存储位置

3. **访问控制**
   - 限制对备份文件的访问
   - 定期轮换数据库密码

## 监控和告警

建议设置以下监控：
- 备份任务成功/失败通知
- 备份文件大小异常告警
- 磁盘空间监控

## 最佳实践

1. **定期测试恢复**
   - 定期验证备份文件可以成功恢复
   - 在测试环境中练习恢复流程

2. **多重备份策略**
   - 保留多个时间点的备份
   - 考虑异地备份存储

3. **文档维护**
   - 记录恢复流程
   - 更新联系信息和访问凭据

## 支持

如有问题，请检查：
1. 日志文件中的错误信息
2. GitHub Actions 运行日志
3. Supabase Dashboard 中的数据库状态

---

**重要提醒**: 在生产环境中恢复数据前，请务必在测试环境中验证备份文件的完整性和恢复流程。