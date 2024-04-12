const std = @import("std");

pub fn Layer(comptime I: usize, comptime O: usize) type {
    const LayerGrads = struct {
        weight_grads: []f64,
        input_grads: []f64,
        const Self = @This();

        pub fn deinit(self: Self, allocator: std.mem.Allocator) void {
            allocator.free(self.weight_grads);
            allocator.free(self.input_grads);
        }
    };

    return struct {
        inputs: usize,
        outputs: usize,
        weights: *[I * O]f64,
        last_inputs: []f64,
        const Self = @This();

        pub fn forward(self: *Self, inputs: []f64, allocator: std.mem.Allocator) ![]f64 {
            const batch_size = inputs.len / I;
            var outputs = try allocator.alloc(f64, batch_size * O);
            var b: usize = 0;
            while (b < batch_size) : (b += 1) {
                var o: usize = 0;
                while (o < O) : (o += 1) {
                    var sum: f64 = 0;
                    var i: usize = 0;
                    while (i < I) : (i += 1) {
                        sum += inputs[b * I + i] * self.weights[O * i + o];
                    }
                    outputs[b * O + o] = sum;
                }
            }
            self.last_inputs = inputs;
            return outputs;
        }

        pub fn backwards(self: *Self, grads: []f64, allocator: std.mem.Allocator) !LayerGrads {
            var weight_grads = try allocator.alloc(f64, I * O);

            const batch_size = self.last_inputs.len / I;
            var input_grads = try allocator.alloc(f64, batch_size * I);

            var b: usize = 0;
            while (b < batch_size) : (b += 1) {
                var i: usize = 0;
                while (i < I) : (i += 1) {
                    var o: usize = 0;
                    while (o < O) : (o += 1) {
                        weight_grads[i * O + o] += (grads[b * O + o] * self.last_inputs[b * I + i]) / @as(f64, @floatFromInt(batch_size));
                        input_grads[b * I + i] += grads[b * O + o] * self.weights[i * O + o];
                    }
                }
            }
            return LayerGrads{ .weight_grads = weight_grads, .input_grads = input_grads };
        }

        pub fn applyGradients(self: *Self, grads: []f64) void {
            var i: usize = 0;
            while (i < I * O) : (i += 1) {
                self.weights[i] -= 0.01 * grads[i];
            }
        }

        pub fn init(allocator: std.mem.Allocator) !Self {
            var memory = try allocator.alloc(f64, I * O);
            var weights = memory[0 .. I * O];
            var prng = std.rand.DefaultPrng.init(123);
            var w: usize = 0;
            while (w < I * O) : (w += 1) {
                weights[w] = prng.random().floatNorm(f64) * 0.2;
            }
            return Self{
                .inputs = I,
                .outputs = O,
                .weights = weights,
                .last_inputs = undefined,
            };
        }

        pub fn deinit(self: *Self, allocator: std.mem.Allocator) void {
            allocator.free(self.weights);
        }
    };
}
