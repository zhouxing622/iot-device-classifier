#!/bin/bash
# 课堂演示 - 捕获特定设备的网络流量
# Classroom Demo - Capture network traffic from specific devices

echo "=============================================="
echo "  🎓 IoT Device Classifier - 课堂演示"
echo "=============================================="
echo ""

# 获取网络接口
INTERFACE="en0"
echo "📡 使用网络接口: $INTERFACE"
echo ""

# 显示本机IP
echo "📍 你的 IP 地址:"
ifconfig $INTERFACE | grep "inet " | awk '{print "   " $2}'
echo ""

# 输入目标IP
echo "请输入要分析的设备 IP 地址"
echo "(多个IP用空格分隔，例如: 192.168.1.100 192.168.1.101)"
echo ""
read -p "🎯 设备 IP: " TARGET_IPS

if [ -z "$TARGET_IPS" ]; then
    echo "❌ 未输入 IP 地址"
    exit 1
fi

# 构建过滤器
FILTER=""
for IP in $TARGET_IPS; do
    if [ -z "$FILTER" ]; then
        FILTER="host $IP"
    else
        FILTER="$FILTER or host $IP"
    fi
done

echo ""
echo "📋 捕获过滤器: $FILTER"
echo ""

# 设置输出文件
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_FILE="$HOME/Desktop/classroom_capture_$TIMESTAMP.pcap"

echo "📁 输出文件: $OUTPUT_FILE"
echo ""

# 开始捕获
PACKET_COUNT=2000
TIMEOUT=30

echo "⏱️  开始捕获 $PACKET_COUNT 个数据包 (最多 ${TIMEOUT}秒)..."
echo "💡 提示: 让同学在手机/电脑上浏览网页或使用应用以产生流量"
echo ""
echo "按 Ctrl+C 可提前停止捕获"
echo ""

# 需要 sudo 权限
sudo tcpdump -i $INTERFACE -c $PACKET_COUNT -w "$OUTPUT_FILE" "$FILTER" 2>&1 &
TCPDUMP_PID=$!

# 显示进度
sleep 2
SECONDS_ELAPSED=0
while kill -0 $TCPDUMP_PID 2>/dev/null && [ $SECONDS_ELAPSED -lt $TIMEOUT ]; do
    PACKETS=$(sudo tcpdump -r "$OUTPUT_FILE" 2>/dev/null | wc -l)
    echo -ne "\r📦 已捕获约 $PACKETS 个数据包... (${SECONDS_ELAPSED}s)"
    sleep 2
    SECONDS_ELAPSED=$((SECONDS_ELAPSED + 2))
done

# 等待 tcpdump 完成
wait $TCPDUMP_PID 2>/dev/null

echo ""
echo ""
echo "=============================================="
echo "✅ 捕获完成!"
echo "=============================================="
echo ""
echo "📁 文件位置: $OUTPUT_FILE"
echo ""
echo "🚀 下一步:"
echo "   1. 打开 Demo: https://iot-device-classifier-josie.streamlit.app/"
echo "   2. 上传文件: $OUTPUT_FILE"
echo "   3. 查看分类结果!"
echo ""

# 询问是否打开 Demo
read -p "是否现在打开 Demo? (y/n): " OPEN_DEMO
if [ "$OPEN_DEMO" = "y" ] || [ "$OPEN_DEMO" = "Y" ]; then
    open "https://iot-device-classifier-josie.streamlit.app/"
fi
