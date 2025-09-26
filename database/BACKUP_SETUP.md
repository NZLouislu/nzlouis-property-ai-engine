# Supabase 数据库备份设置说明

## 当前状态
备份脚本已修复语法错误，现在可以正常运行。但是需要正确配置 `DATABASE_URL` 才能进行完整的 pg_dump 备份。

## 备份类型

### 1. JSON 数据备份（当前可用）
- ✅ 使用 Supabase API 导出所有表数据
- ✅ 包含所有记录的 JSON 格式文件
- ✅ 适合数据分析和检查
- ⚠️ 不包含数据库结构（表结构、索引、约束等）

### 2. 完整数据库备份（需要 DATABASE_URL）
- 🔧 使用 pg_dump 创建完整备份
- 🔧 包含数据库结构和数据
- 🔧 可以完全恢复数据库
- 🔧 需要配置 DATABASE_URL

## 如何获取 DATABASE_URL

### 方法1：从 Supabase 控制台获取
1. 登录 Supabase 控制台
2. 选择你的项目
3. 进入 Settings → Database
4. 在 "Connection string" 部分找到 "URI" 格式的连接字符串
5. 格式类似：`postgresql://postgres:[password]@db.[project-ref].supabase.co:5432/postgres`

### 方法2：构建 DATABASE_URL
如果你有以下信息：
- Project Reference（从 SUPABASE_URL 中提取）
- Database Password

格式：`postgresql://postgres:[password]@db.[project-ref].supabase.co:5432/postgres`

## GitHub Secrets 配置

在 GitHub 仓库中添加以下 Secrets：

### 必需的 Secrets：
- `SUPABASE_URL` - 你的 Supabase 项目 URL
- `SUPABASE_KEY` - 你的 Supabase API Key

### 可选的 Secrets（用于完整备份）：
- `DATABASE_URL` - 完整的 PostgreSQL 连接字符串
- `SUPABASE_DB_PASSWORD` - 数据库密码（如果没有 DATABASE_URL）

## 当前备份功能

即使没有 `DATABASE_URL`，备份脚本仍然会：

1. ✅ 创建 JSON 数据备份（包含所有表数据）
2. ✅ 创建备份元数据文件
3. ✅ 更新备份状态
4. ✅ 上传到 GitHub Artifacts（保留90天）

## 备份文件说明

### JSON 备份文件
- 文件名：`supabase_data_backup_YYYYMMDD_HHMMSS.json`
- 内容：所有表的数据
- 格式：
```json
{
  "backup_timestamp": "2025-09-27T11:30:00.000Z",
  "backup_type": "json_api_export",
  "tables": {
    "properties": {
      "record_count": 1234,
      "data": [...]
    },
    "property_history": {
      "record_count": 5678,
      "data": [...]
    }
  },
  "summary": {
    "total_tables": 5,
    "successful_tables": 5,
    "total_records": 12345
  }
}
```

### 元数据文件
- 文件名：`backup_metadata_YYYYMMDD_HHMMSS.json`
- 内容：备份信息和恢复说明

## 下一步建议

1. **立即可用**：当前配置已经可以创建有用的 JSON 数据备份
2. **完整备份**：如需完整的数据库备份，请在 GitHub Secrets 中添加 `DATABASE_URL`
3. **测试备份**：可以手动触发工作流测试备份功能

## 手动触发备份

在 GitHub Actions 页面：
1. 选择 "Weekly Supabase Database Backup" 工作流
2. 点击 "Run workflow"
3. 选择分支并运行

备份将保存在 GitHub Artifacts 中，保留90天。