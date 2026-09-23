# frozen_string_literal: true

module NArrayLLM
  module BinaryIO
    module_function

    def read_exactly(io, nbytes, path, what)
      raw = io.read(nbytes)
      if raw.nil? || raw.bytesize != nbytes
        raise FormatError, "#{path}: truncated while reading #{what}: " \
                           "wanted #{nbytes} bytes, got #{raw ? raw.bytesize : 0}"
      end

      raw
    end

    def read_header(io, path, magic, version, ints: 256)
      header = read_exactly(io, ints * 4, path, 'header').unpack('l<*')
      unless header[0] == magic
        raise FormatError, "#{path}: bad magic #{header[0]}, expected #{magic}"
      end
      unless header[1] == version
        raise FormatError, "#{path}: unsupported version #{header[1]}, expected #{version}"
      end

      header
    end
  end
end
