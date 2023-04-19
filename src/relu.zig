const std = @import("std");

pub const Relu = struct {
    last_inputs: []f64,
    const Self = @This();

    pub fn new() Self {
        return Self {
            .last_inputs = undefined,
        };
    }

    pub fn forward(self: *Self, inputs: []f64, allocator: *std.mem.Allocator) ![]f64 {
        var outputs = try allocator.alloc(f64, inputs.len);
        var i: usize = 0;
        while (i < inputs.len): (i += 1) {
            if (inputs[i] < 0) {
                outputs[i] = 0;
            } else {
                outputs[i] = inputs[i];
            }
        }
        self.last_inputs = inputs;
        return outputs;
    }

    pub fn backwards(self: *Self, grads: []f64, allocator: *std.mem.Allocator) ![]f64 {
        var outputs = try allocator.alloc(f64, grads.len);
        var i: usize = 0;
        while (i < self.last_inputs.len): (i += 1) {
            if (self.last_inputs[i] < 0) {
                grads[i] = 0;
            } else {
                outputs[i] = grads[i];
            }
        }
        return outputs;
    }
};
