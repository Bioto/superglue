/**
 * gRPC client helpers generated from proto/superglue.proto.
 *
 * Regenerate Python stubs: scripts/generate-proto-clients.sh
 * Rust tonic client: cargo build --features grpc
 */
import path from "node:path";
import { fileURLToPath } from "node:url";
import * as grpc from "@grpc/grpc-js";
import * as protoLoader from "@grpc/proto-loader";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const PROTO = path.resolve(__dirname, "../../proto/superglue.proto");

const packageDefinition = protoLoader.loadSync(PROTO, {
  keepCase: false,
  longs: String,
  enums: String,
  defaults: true,
  oneofs: true,
});

export const superglueProto = grpc.loadPackageDefinition(
  packageDefinition,
) as unknown as {
  superglue: {
    SuperglueService: grpc.ServiceClientConstructor;
  };
};

/** Connect a gRPC client to a Superglue sidecar (Complete / Stream / Responses RPCs). */
export function connectSuperglue(
  address: string,
  credentials: grpc.ChannelCredentials = grpc.credentials.createInsecure(),
) {
  const Service = superglueProto.superglue.SuperglueService;
  return new Service(address, credentials);
}

export type SuperglueClient = ReturnType<typeof connectSuperglue>;
