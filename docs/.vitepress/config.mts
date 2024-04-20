import { defineConfig } from "vitepress";
import zhConfig from "./themeConfig.zh.mts";
import enConfig from "./themeConfig.en.mts";

import mathjax3 from "markdown-it-mathjax3";

// https://vitepress.dev/reference/site-config
export default defineConfig({
  markdown: {
    config: (md) => {
      md.use(mathjax3);
    },
  },
  // 由于vitepress编译生成静态html文件时，无法识别插件生成的特殊标签，故需在编译时进行处理，将特殊标签定位自定义标签，防止编译报错
  vue: {
    template: {
      compilerOptions: {
        isCustomElement: (tag) => ["mjx-container"].includes(tag),
      },
    },
  },

  title: "Aka",
  description: "",
  lang: "zh_CN",
  themeConfig: {
    search: {
      provider: 'local',
      options: {
        locales: {
          zh: {
            translations: {
              button: {
                buttonText: '搜索文档',
                buttonAriaLabel: '搜索文档'
              },
              modal: {
                noResultsText: '无法找到相关结果',
                resetButtonTitle: '清除查询条件',
                footer: {
                  selectText: '选择',
                  navigateText: '切换'
                }
              }
            }
          }
        }
      }
    }
  },
  /* 语言设置 */
  locales: {
    root: { label: "简体中文", lang: "zh-CN", link: "/", ...zhConfig},
    //en: { label: "English", lang: "en-US", link: "/en/", ...enConfig}
  },
});
