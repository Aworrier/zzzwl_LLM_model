#!/bin/bash

# 检查是否传入了 commit 信息
if [ -z "$1" ]; then
    echo "错误：请在命令后添加 commit 信息，例如版本号"
    echo "用法：./git-auto-push.sh \"v1.2.3 提交说明\""
    exit 1
fi

# 显示即将执行的操作
echo "📁 添加更改中..."
git add .

echo "📝 提交更改：$1"
git commit -m "$1"

echo "🚀 推送到远程仓库..."
git push

echo "✅ 提交完成！"
