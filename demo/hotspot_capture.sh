#!/bin/bash
# 热点模式 - 捕获连接到你热点的设备流量
# Hotspot Mode - Capture traffic from devices connected to your hotspot

clear
echo "=============================================="
echo "  🔥 IoT Device Classifier - 热点捕获模式"
echo "=============================================="
echo ""

# 检查是否已开启热点
echo "📋 第一步：请确保已开启 Mac 热点"
echo ""
echo "   开启方法："
echo "   1. 打开 系统设置 → 通用 → 共享"
echo "   2. 点击 '互联网共享'"
echo "   3. '共享以下来源的连接': 选择你的网络 (如 Wi-Fi 或以太网)"
echo "   4. '用以下端口共享给电脑': 勾选 'Wi-Fi'"
echo "   5. 点击 Wi-Fi 选项，设置热点名称和密码"
echo "   6. 打开 '互联网共享' 开关"
echo ""

read -p "热点已开启? (y/n): " HOTSPOT_READY
if [ "$HOTSPOT_READY" != "y" ] && [ "$HOTSPOT_READY" != "Y" ]; then
    echo ""
    echo "请先开启热点，然后重新运行此脚本。"
    exit 1
fi

echo ""
echo "=============================================="
echo "  🔍 检测热点网络接口..."
echo "=============================================="
echo ""

# 查找热点接口 (通常是 bridge100 或类似)
BRIDGE_INTERFACE=$(ifconfig | grep -E "^bridge[0-9]+" | head -1 | cut -d: -f1)

if [ -z "$BRIDGE_INTERFACE" ]; then
    echo "⚠️  未检测到热点桥接接口，尝试使用 ap1..."
    BRIDGE_INTERFACE="ap1"
fi

# 显示所有网络接口供选择
echo "检测到的网络接口:"
echo ""
ifconfig | grep -E "^[a-z]+[0-9]+:" | cut -d: -f1 | while read iface; do
    IP=$(ifconfig $iface 2>/dev/null | grep "inet " | awk '{print $2}')
    if [ -n "$IP" ]; then
        echo "   $iface: $IP"
    fi
done

echo ""
echo "🔥 推测的热点接口: $BRIDGE_INTERFACE"
echo ""

read -p "使用此接口? 或输入其他接口名称 [$BRIDGE_INTERFACE]: " USER_INTERFACE
if [ -n "$USER_INTERFACE" ]; then
    BRIDGE_INTERFACE=$USER_INTERFACE
fi

# 获取热点IP
HOTSPOT_IP=$(ifconfig $BRIDGE_INTERFACE 2>/dev/null | grep "inet " | awk '{print $2}')

if [ -z "$HOTSPOT_IP" ]; then
    echo ""
    echo "⚠️  无法获取接口 $BRIDGE_INTERFACE 的 IP 地址"
    echo "请确保热点已正确开启，或手动指定接口。"
    exit 1
fi

echo ""
echo "=============================================="
echo "  📡 热点信息"
echo "=============================================="
echo ""
echo "   接口: $BRIDGE_INTERFACE"
echo "   热点 IP: $HOTSPOT_IP"
echo ""

# 等待设备连接
echo "=============================================="
echo "  📱 第二步：让同学连接你的热点"
echo "=============================================="
echo ""
echo "   告诉同学："
echo "   1. 打开 Wi-Fi 设置"
echo "   2. 连接到你的热点网络"
echo "   3. 连接后，在设备上浏览网页"
echo ""

read -p "有设备已连接? (y/n): " DEVICES_CONNECTED
if [ "$DEVICES_CONNECTED" != "y" ] && [ "$DEVICES_CONNECTED" != "Y" ]; then
    echo ""
    echo "请等待设备连接后重新运行。"
    exit 1
fi

# 显示已连接的设备
echo ""
echo "🔍 尝试检测已连接的设备..."
echo ""

# 使用 arp 扫描同网段设备
arp -a | grep $BRIDGE_INTERFACE 2>/dev/null

echo ""
echo "=============================================="
echo "  🎬 第三步：开始捕获流量"
echo "=============================================="
echo ""

# 设置输出文件
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_FILE="$HOME/Desktop/hotspot_capture_$TIMESTAMP.pcap"

PACKET_COUNT=3000
TIMEOUT=60

echo "📁 输出文件: $OUTPUT_FILE"
echo "📦 目标数据包: $PACKET_COUNT"
echo "⏱️  超时时间: ${TIMEOUT}秒"
echo ""
echo "💡 提示: 让同学们在手机上："
echo "   - 打开网页浏览"
echo "   - 刷新社交媒体"
echo "   - 播放视频"
echo "   - 发送消息"
echo ""

read -p "准备好了? 按 Enter 开始捕获..."

echo ""
echo "🔴 正在捕获... (按 Ctrl+C 提前停止)"
echo ""

# 开始捕获
sudo tcpdump -i $BRIDGE_INTERFACE -c $PACKET_COUNT -w "$OUTPUT_FILE" 2>&1 &
TCPDUMP_PID=$!

# 显示进度
SECONDS_ELAPSED=0
while kill -0 $TCPDUMP_PID 2>/dev/null && [ $SECONDS_ELAPSED -lt $TIMEOUT ]; do
    if [ -f "$OUTPUT_FILE" ]; then
        SIZE=$(ls -lh "$OUTPUT_FILE" 2>/dev/null | awk '{print $5}')
        echo -ne "\r📦 已捕获: $SIZE ... (${SECONDS_ELAPSED}s / ${TIMEOUT}s)   "
    fi
    sleep 2
    SECONDS_ELAPSED=$((SECONDS_ELAPSED + 2))
done

# 停止捕获
sudo kill $TCPDUMP_PID 2>/dev/null
wait $TCPDUMP_PID 2>/dev/null

echo ""
echo ""

# 显示捕获结果
if [ -f "$OUTPUT_FILE" ]; then
    FINAL_SIZE=$(ls -lh "$OUTPUT_FILE" | awk '{print $5}')
    PACKET_CAPTURED=$(sudo tcpdump -r "$OUTPUT_FILE" 2>/dev/null | wc -l | tr -d ' ')
    
    echo "=============================================="
    echo "  ✅ 捕获完成!"
    echo "=============================================="
    echo ""
    echo "   📁 文件: $OUTPUT_FILE"
    echo "   📦 数据包: $PACKET_CAPTURED"
    echo "   💾 文件大小: $FINAL_SIZE"
    echo ""
    echo "=============================================="
    echo "  🚀 第四步：上传到 Demo 分析"
    echo "=============================================="
    echo ""
    echo "   1. 打开: https://iot-device-classifier-josie.streamlit.app/"
    echo "   2. 上传: $OUTPUT_FILE"
    echo "   3. 查看分类结果!"
    echo ""
    
    read -p "是否现在打开 Demo? (y/n): " OPEN_DEMO
    if [ "$OPEN_DEMO" = "y" ] || [ "$OPEN_DEMO" = "Y" ]; then
        open "https://iot-device-classifier-josie.streamlit.app/"
        echo ""
        echo "🌐 已打开 Demo，请上传文件进行分析！"
    fi
else
    echo "❌ 捕获失败，未生成文件。"
    echo "请检查热点是否正确开启，是否有设备连接。"
fi

echo ""
echo "=============================================="
echo "  🎓 演示话术建议"
echo "=============================================="
echo ""
echo '  "现在我捕获了大家手机连接我热点时产生的网络流量。'
echo '   让我们用机器学习模型来分析这些流量...'
echo '   看！模型识别出了 X 个 smartphone 设备，'
echo '   置信度高达 XX%！这说明我们的模型确实能够'
echo '   通过网络流量特征识别出设备类型。"'
echo ""
