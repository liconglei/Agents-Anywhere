# DSH Bridge Next 2.0.8

`@agents-anywhere/dsh-bridge-next` 2.0.8 是面向 Server 2.0.x 的 DSH 插件维护版。它不要求 Server 升级到 2.1.0；NAS 上的 Server 2.0.3 可以继续使用。

## 修复

- DSH 0.2.1 交互流 owner peer 授权：打开 `$events` 和回复 `$events/result` 时使用 `browserPeer` / `operator`，修复提问、审批卡片不弹或回复失败。
- Connector 侧按 `session.interaction.approval` 上报 DSH 交互能力，并上报 session notices。
- 内嵌 Connector 2.0.7，包含 OpenCode 提问的 `inputRequest` 表单修复。

## 兼容性

- Server：2.0.x；不要求 Server 2.1.0。Server 2.0.3 上使用本包，不要混装 2.1.0 DSH 插件。
- DSH：`>=0.2.0-rc.1 <0.3.0-0`。
- 2.1.0 组件组合需要 Server 2.1.0；本包不是 2.1.0 的替代。

## 产物

| 文件 | 大小 | SHA-256 |
| --- | ---: | --- |
| `agents-anywhere-dsh-bridge-next-2.0.8.tgz` | 893,022 字节（222 个文件） | `05cf1c37358fe616c621018f6cbd2facb9df24daf08d40a8f0fa0a247185a1a5` |

## 验证

- DSH 交互流定向测试 `tests/unit/interaction-stream.test.ts`：6/6 通过。
- 内嵌 Connector 的 OpenCode 定向测试：135 passed / 2 skipped。
- 本机完整 `prepack` 的测试阶段被 Windows 临时目录 ACL 和回环端口占用阻断；`typecheck`、`build`、`check:build` 已通过，tgz 由该构建产物生成。

安装：

```bash
dsh plugin --profile <PROFILE> add ./agents-anywhere-dsh-bridge-next-2.0.8.tgz
```

来源：`release/connector-2.0.7-opencode-form`，提交 `197a4c4bacd92221dda7f7fd8dc593773adc4e5a`（基线 `20ab3453`，含 `45c909dd`、`a9c182c1`、`197a4c4b`）。
