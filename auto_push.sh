#!/bin/bash

# 提示用户输入提交信息
echo "请输入提交信息:"
read commit_message

# 检查提交信息是否为空
if [ -z "$commit_message" ]; then
    echo "错误: 提交信息不能为空"
    exit 1
fi

# 添加所有更改
echo "正在添加更改..."
git add .

# 提交更改
echo "正在提交更改，提交信息: $commit_message"
git commit -m "$commit_message"

# 推送到远程仓库
echo "正在推送到远程仓库..."
current_branch=$(git rev-parse --abbrev-ref HEAD)
git push origin $current_branch

echo "推送完成!"