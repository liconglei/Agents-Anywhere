# Connector 2.0.7

Connector 2.0.7 是面向 Server 2.0.x 的维护版，适用于当前 NAS 的 Server 2.0.3。它不要求 Server 升级到 2.1.0；Connector 2.1.0 与 Server 2.0.3 不是受支持组合。

## 修复

- OpenCode 提问改用 `inputRequest` 表单上报，结构化答案映射回 OpenCode 字段。
- DSH 侧按 `session.interaction.approval` 上报交互能力，并上报 session notices。
- 包含 DSH 会话同步增量与重订阅防抖修复。

## 兼容性

- Server：2.0.x（当前 NAS 2.0.3 可用）。
- 不适用于 Server 2.0.3 + Connector 2.1.0 组合；2.1.0 需要 Server 2.1.0。
- 本次不发布 APK。

## 产物

| 文件 | 平台 | 大小 | SHA-256 |
| --- | --- | ---: | --- |
| `agents-anywhere-connector-2.0.7-win-x64.zip` | Windows x64 | 49,590,366 | `e3189160bb72be4602ce1774b6f70969b82998e94712a90d322e7e343444ef03` |
| `agents-anywhere-connector-2.0.7-win-arm64.zip` | Windows arm64 | 46,799,933 | `3c2c5a1ec5cd82cdf22499c789a063e0eaf1340421f06c63336da5626ce00e29` |
| `agents-anywhere-connector-2.0.7-linux-x64.tar.gz` | Linux x64 | 61,083,046 | `1a624a916ffb7b7eada781ea0767f159a5cbfc19e4364d4e3a8f508d556f0453` |
| `agents-anywhere-connector-2.0.7-linux-arm64.tar.gz` | Linux arm64 | 53,856,735 | `b911a590a115349a594cfc3dc7c76c1393fa1bc8a3d43ff41573465174d38737` |

来源：`release/connector-2.0.7-opencode-form`，提交 `197a4c4bacd92221dda7f7fd8dc593773adc4e5a`（版本基线 `20ab3453`）。产物由 `connector/scripts/package.mjs` 生成。
