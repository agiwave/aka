import type { DefaultTheme, LocaleSpecificConfig } from "vitepress";

const themeConfig: LocaleSpecificConfig<DefaultTheme.Config> = {
  // https://vitepress.dev/reference/default-theme-config
  themeConfig: {
    nav: [
      {
        text: "Docs",
        items: [{ text: "Neural Network", link: "/nn/preface/background/" }],
      },
    ],

    sidebar: [
      {
        text: "Dirs",
        items: [
          {
            text: "1、前言",
            items: [
              { text: "1.1、背景", link: "/nn/preface/background/" },
              { text: "1.2、简介", link: "/nn/preface/intro/" },
            ],
          },
          {
            text: "2、神经网络原理",
            items: [
              { text: "2.1、总体架构", link: "/nn/principle/arch/" },
              { text: "2.2、输入输出", link: "/nn/principle/io/" },
              { text: "2.3、网络结构", link: "/nn/principle/structure/" },
              { text: "2.4、学习原理", link: "/nn/principle/learn/" },
              { text: "2.5、小结", link: "/nn/principle/summarize/" },
            ],
          },
          {
            text: "3、经典神经网络",
            items: [
              { text: "3.1、简介", link: "/nn/classic/intro/" },
              { text: "3.2、多层感知机(MLP)", link: "/nn/classic/mlp/" },
              { text: "3.3、卷积神经网络(CNN)", link: "/nn/classic/cnn/" },
              { text: "3.4、循环神经网络(RNN)", link: "/nn/classic/rnn/" },
              { text: "3.5、Transformer", link: "/nn/classic/transformer/" },
              { text: "3.6、小结", link: "/nn/classic/summarize/" },
            ],
          },
        ],
      },
    ],
    outline: {
      level: "deep",
    },

    footer: {
      message: "Contact: sanbgi@qq.com",
      copyright: "Copyright © 2023-present",
    },
  },
};

export default themeConfig;
