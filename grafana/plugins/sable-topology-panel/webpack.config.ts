import type { Configuration } from 'webpack';
import * as path from 'path';
import CopyWebpackPlugin from 'copy-webpack-plugin';

const config = (env: { production?: boolean }): Configuration => ({
  mode: env.production ? 'production' : 'development',
  devtool: env.production ? 'source-map' : 'eval-source-map',
  entry: './src/module.ts',
  output: {
    filename: 'module.js',
    path: path.resolve(__dirname, 'dist'),
    libraryTarget: 'amd',
    clean: true,
  },
  externals: [
    'react',
    'react-dom',
    '@grafana/data',
    '@grafana/ui',
    '@grafana/runtime',
    function ({ request }: any, callback: any) {
      if (request && /^@grafana\//.test(request)) {
        return callback(null, 'amd ' + request);
      }
      callback();
    },
  ] as any,
  resolve: {
    extensions: ['.ts', '.tsx', '.js'],
  },
  module: {
    rules: [
      {
        test: /\.tsx?$/,
        use: 'ts-loader',
        exclude: /node_modules/,
      },
      {
        test: /\.css$/,
        use: ['style-loader', 'css-loader'],
      },
    ],
  },
  plugins: [
    new CopyWebpackPlugin({
      patterns: [{ from: 'plugin.json', to: '.' }],
    }),
  ],
});

export default config;
