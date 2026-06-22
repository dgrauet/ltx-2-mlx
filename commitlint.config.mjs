export default {
  extends: ['@commitlint/config-conventional'],
  rules: {
    // Dependabot's grouped PRs put long markdown tables + URLs in the commit
    // body and footer; don't reject those on line length. Subject/header rules
    // still apply.
    'body-max-line-length': [0],
    'footer-max-line-length': [0],
  },
};
