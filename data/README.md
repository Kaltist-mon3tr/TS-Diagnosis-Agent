# Manifest 数据格式

每行一个 JSON 对象，至少包含：

- sample_id：唯一窗口 ID；
- path：本地 .npy 路径，数组形状为 [channels, time] 或 [time]；
- domain：源域名称或 target；
- label：训练或评估时的 0/1 标签；无标签池可省略，阶段二不会读取它。

可选字段为 start、length、condition、record_id、machine_id 和 group_id。跨集合去重会检查 sample_id、同一 .npy 文件的重叠区间，以及显式提供的 group_id。若要按设备或完整记录隔离，应把对应的设备/记录标识填入所有窗口的 group_id 字段，然后再切窗，避免相邻片段泄漏。

阶段二的 target_calibration 只能含正常目标域窗口，并独立于 target_normal；它只用于模型选择和部署阈值。

