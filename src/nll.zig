const std = @import("std");

pub fn NLL(comptime I: usize) type {
    const NLLOuput = struct {
        loss: []f64,
        input_grads: []f64,
        const Self = @This();

        pub fn deinit(self: Self, allocator: std.mem.Allocator) void {
            allocator.free(self.loss);
            allocator.free(self.input_grads);
        }
    };

    return struct {
        pub fn nll(inputs: []f64, targets: []u8, allocator: std.mem.Allocator) !NLLOuput {
            const batch_size = targets.len;
            var sum_e = try allocator.alloc(f64, batch_size);
            defer allocator.free(sum_e);
            var b: usize = 0;
            while (b < batch_size) : (b += 1) {
                var sum: f64 = 0;
                var i: usize = 0;
                while (i < I) : (i += 1) {
                    sum += std.math.exp(inputs[b * I + i]);
                }
                sum_e[b] = sum;
            }

            var loss = try allocator.alloc(f64, batch_size);
            b = 0;
            while (b < batch_size) : (b += 1) {
                loss[b] = -1 * @log(std.math.exp(inputs[b * I + targets[b]]) / sum_e[b]);
            }

            var input_grads = try allocator.alloc(f64, batch_size * I);
            b = 0;
            while (b < batch_size) : (b += 1) {
                var i: usize = 0;
                while (i < I) : (i += 1) {
                    input_grads[b * I + i] = std.math.exp(inputs[b * I + i]) / sum_e[b];
                    if (i == targets[b]) {
                        input_grads[b * I + i] -= 1;
                    }
                }
            }

            return NLLOuput{ .loss = loss, .input_grads = input_grads };
        }
    };
}
