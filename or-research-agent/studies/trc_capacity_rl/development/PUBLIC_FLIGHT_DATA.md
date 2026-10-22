# 已取得的公开无人机测量数据

原始仓库：[CMU KiltHub, Data Collected with Package Delivery Quadcopter Drone](https://kilthub.cmu.edu/articles/dataset/Data_Collected_with_Package_Delivery_Quadcopter_Drone/12683453)，DOI 10.1184/R1/12683453.v1；官方API确认CC BY 4.0。[论文预印本](https://arxiv.org/abs/2103.13313)说明数据涉及209次记录、约10小时45分钟和约65公里。该数据可补充LaDe缺乏飞行测量的问题。

已下载 parameters.csv、README.txt、flights.zip；全部与官方MD5一致，另记录SHA256在 `data/drone_measurement_manifest.json`。压缩包约41.3MB，含209份飞行CSV及一个目录。下载脚本为 `fetch_drone_data.py`，原始文件保留在本地，不随本项目发布。

元数据含设定速度、载荷、飞行高度、日期、时间与航线；传感数据含位置、速度、风相关读数及电压电流。可用于检查时间/能耗尺度、建立单次飞行残差分布。还未拟合校准模型，也不声称已支持当前合成倍率或充电时长。

限制与待核对：这是单平台逐次采集，不能直接识别同一时刻多架无人机的相关扰动或站点充电拥堵。论文、参数表和README的日期描述存在待核对之处：参数表首条显示2019-04-07，README末尾范围写20190704–20191024；不得擅自按某一解释建立时间切分。需核对原日志日期和作者处理代码，并阅读传感器偏差说明；不是把字段wind_speed直接当精确环境风速。

2026-10-04已完成全部209份日志的字段、单位、时间轴及积分量审计。完整路线记录196份，中位日志时长184秒、水平速度积分路径458米、电能20.73Wh。完整飞行路径积分总长与论文约65公里不符，差异保留为未解释；没有强制缩放或筛除记录。详见[第三轮结果](RESULTS_03.md)与 `results/flight_audit_001`。下一步需要飞行阶段和航线定义核对；尚未拟合校准模型。不能用实测单机数据给全相关天气或现实运营收益背书。
