import type { DefaultTheme, LocaleSpecificConfig } from 'vitepress'

const themeConfig: LocaleSpecificConfig<DefaultTheme.Config> = {
  
  // https://vitepress.dev/reference/default-theme-config
  themeConfig: {
    nav: [
      {
        text: "文档",
        items: [{ text: "工具介绍", link: "/intro/" }],
      },
    ],

    lightModeSwitchTitle: "切换到浅色模式",
    darkModeSwitchTitle: "切换到深色模式",

    sidebar: [
      {
        text: "",
        items: [
          { text: "简介", link: "/intro/" },
          // {
          //   text: "2、神经网络原理",
          //   items: [
          //     { text: "2.1、简介", link: "/nn/principle/intro/" },
          //     { text: "2.2、数据编码", link: "/nn/principle/encode/" },
          //     { text: "2.3、网络结构", link: "/nn/principle/structure/" },
          //     { text: "2.4、学习原理", link: "/nn/principle/learn/" },
          //     { text: "2.5、工程实践", link: "/nn/principle/practice/" },
          //     { text: "2.6、小结", link: "/nn/principle/summarize/" },
          //   ],
          // },
          // {
          //   text: "3、经典神经网络",
          //   items: [
          //     { text: "3.1、简介", link: "/nn/classic/intro/" },
          //     { text: "3.2、多层感知机(MLP)", link: "/nn/classic/mlp/" },
          //     { text: "3.3、卷积神经网络(CNN)", link: "/nn/classic/cnn/" },
          //     { text: "3.4、循环神经网络(RNN)", link: "/nn/classic/rnn/" },
          //     { text: "3.5、Transformer", link: "/nn/classic/transformer/" },
          //     { text: "3.6、小结", link: "/nn/classic/summarize/" },
          //   ],
          // },
          // {
          //   text: "4、神经网络历史",
          //   items: [
          //     { text: "4.1、简介", link: "/nn/history/00intro/" },
          //     { text: "4.2、第一次浪潮", link: "/nn/history/01wave/" },
          //     { text: "4.3、第一次低谷", link: "/nn/history/02trough/" },
          //     { text: "4.4、第二次浪潮", link: "/nn/history/03wave/" },
          //     { text: "4.5、第二次低谷", link: "/nn/history/04trough/" },
          //     { text: "4.6、第三次浪潮", link: "/nn/history/05wave/" },
          //   ],
          // },
        ],
      },
    ],
    outline: {
      level: "deep",
      label: "大纲",
    },
    docFooter: {
      prev: "上一页",
      next: "下一页",
    },
    footer: {
      message: '联系方式: sanbgi@qq.com',
      copyright: 'Copyright © 2023-present'
    }
  }
};

export default themeConfig;