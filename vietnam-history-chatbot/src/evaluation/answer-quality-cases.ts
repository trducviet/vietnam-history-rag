/**
 * Answer Quality Test Cases — 20 structured test cases for evaluating
 * end-to-end answer quality across multiple question categories.
 *
 * PATCH 6: New file.
 */

// ─── Types ───────────────────────────────────────────────────

export type AnswerCategory =
  | 'fact'
  | 'date'
  | 'actor'
  | 'location'
  | 'explanation'
  | 'comparison'
  | 'timeline'
  | 'disambiguation'
  | 'planned_not_executed'
  | 'out_of_scope';

export interface AnswerQualityCase {
  id: string;
  query: string;

  category: AnswerCategory;
  difficulty: 'easy' | 'medium' | 'hard';

  /** ALL of these IDs must appear in citations/related_events */
  expectedCitationIds?: string[];
  /** At least ONE of these IDs must appear */
  expectedAnyCitationIds?: string[];
  /** NONE of these should appear in primary citations */
  forbiddenCitationIds?: string[];

  /** ALL of these terms must appear in answer+explanation */
  requiredTerms?: string[];
  /** For each group, at least ONE term must appear */
  requiredAnyTerms?: string[][];
  /** NONE of these terms should appear */
  forbiddenTerms?: string[];

  /** Response should mention planned/not executed status */
  shouldMentionPlannedNotExecuted?: boolean;
  /** Response should refuse or limit scope */
  shouldRefuseOrLimitScope?: boolean;

  notes?: string;
}

// ─── Test Cases ──────────────────────────────────────────────

export const ANSWER_QUALITY_CASES: AnswerQualityCase[] = [
  // 1. Fact/date — Điện Biên Phủ start date
  {
    id: 'aq_fact_001',
    query: 'Chiến dịch Điện Biên Phủ bắt đầu vào ngày nào?',
    category: 'fact',
    difficulty: 'easy',
    expectedCitationIds: ['EVT_0207'],
    requiredTerms: ['Điện Biên Phủ'],
    requiredAnyTerms: [['13-3-1954', '13/3/1954', '13 tháng 3']],
  },

  // 2. Clause / planned — Geneva elections
  {
    id: 'aq_clause_001',
    query: 'Điều khoản nào của Hiệp định Genève dự kiến tổng tuyển cử thống nhất vào năm 1956?',
    category: 'planned_not_executed',
    difficulty: 'medium',
    expectedCitationIds: ['EVT_0219'],
    expectedAnyCitationIds: ['EVT_0211'],
    requiredTerms: ['tổng tuyển cử'],
    requiredAnyTerms: [['1956']],
    shouldMentionPlannedNotExecuted: true,
  },

  // 3. Actor — Declaration of Independence
  {
    id: 'aq_actor_001',
    query: 'Ai đọc Tuyên ngôn Độc lập ngày 2-9-1945 tại Ba Đình?',
    category: 'actor',
    difficulty: 'easy',
    expectedCitationIds: ['EVT_0130'],
    requiredAnyTerms: [['Hồ Chí Minh', 'Chủ tịch Hồ Chí Minh']],
  },

  // 4. Treaty — Geneva provisions
  {
    id: 'aq_treaty_001',
    query: 'Hiệp định Genève quy định gì về Việt Nam?',
    category: 'explanation',
    difficulty: 'medium',
    expectedAnyCitationIds: ['EVT_0211', 'EVT_0212', 'EVT_0219'],
    requiredAnyTerms: [
      ['vĩ tuyến 17', 'ranh giới tạm thời', 'giới tuyến'],
      ['tổng tuyển cử', '1956'],
    ],
  },

  // 5. Explanation — Geneva significance
  {
    id: 'aq_explain_001',
    query: 'Hiệp định Genève có ý nghĩa gì đối với việc chia cắt Việt Nam?',
    category: 'explanation',
    difficulty: 'medium',
    expectedAnyCitationIds: ['SYN_TREATY_008', 'EVT_0212', 'EVT_0219'],
    requiredAnyTerms: [
      ['vĩ tuyến 17', 'ranh giới tạm thời', 'giới tuyến'],
      ['tạm thời', 'chia cắt'],
    ],
  },

  // 6. Comparison — DBP victory vs opening
  {
    id: 'aq_compare_001',
    query: 'Chiến thắng Điện Biên Phủ khác với việc mở màn chiến dịch Điện Biên Phủ như thế nào?',
    category: 'comparison',
    difficulty: 'hard',
    expectedAnyCitationIds: ['SYN_COMPARE_003', 'EVT_0207', 'EVT_0209'],
    requiredAnyTerms: [
      ['bắt đầu', 'mở màn', 'khai mào'],
      ['kết thúc', 'chiến thắng', 'thắng lợi'],
    ],
  },

  // 7. Disambiguation — US withdrawal vs Paris signing
  {
    id: 'aq_disambig_001',
    query: 'Sự kiện nào nói về việc Mỹ rút quân, khác với việc ký Hiệp định Paris?',
    category: 'disambiguation',
    difficulty: 'hard',
    expectedCitationIds: ['EVT_0339'],
    forbiddenCitationIds: ['EVT_0337'],
    requiredAnyTerms: [['rút quân', 'Mỹ rút']],
  },

  // 8. Modern date — ASEAN
  {
    id: 'aq_date_001',
    query: 'Việt Nam gia nhập ASEAN khi nào?',
    category: 'date',
    difficulty: 'easy',
    expectedAnyCitationIds: ['EVT_0416', 'EVT_0397'],
    requiredTerms: ['ASEAN'],
    requiredAnyTerms: [['1995']],
  },

  // 9. Organization — MTDTGP
  {
    id: 'aq_org_001',
    query: 'Tổ chức nào được thành lập ngày 20-12-1960 để quy tụ các lực lượng chống Mỹ ở miền Nam?',
    category: 'fact',
    difficulty: 'medium',
    expectedCitationIds: ['EVT_0257'],
    requiredAnyTerms: [
      ['Mặt trận Dân tộc Giải phóng miền Nam', 'Mặt trận Dân tộc Giải phóng', 'MTDTGPMN'],
    ],
  },

  // 10. Out of scope — Trần dynasty
  {
    id: 'aq_oos_001',
    query: 'Nhà Trần thành lập năm nào?',
    category: 'out_of_scope',
    difficulty: 'medium',
    shouldRefuseOrLimitScope: true,
    requiredAnyTerms: [
      ['ngoài phạm vi', '1858', '2000', 'không nằm trong', 'không có trong', 'ngoài khoảng thời gian'],
    ],
  },

  // 11. Out of scope — WTO
  {
    id: 'aq_oos_002',
    query: 'Việt Nam gia nhập WTO năm nào?',
    category: 'out_of_scope',
    difficulty: 'medium',
    shouldRefuseOrLimitScope: true,
    requiredAnyTerms: [
      ['ngoài phạm vi', '2000', 'không nằm trong', 'không có trong', 'ngoài khoảng thời gian', '2007'],
    ],
    notes: 'WTO là 2007, ngoài phạm vi 1858–2000 của project.',
  },

  // 12. Timeline — events leading to Geneva
  {
    id: 'aq_timeline_001',
    query: 'Nêu các mốc chính dẫn tới Hiệp định Genève 1954.',
    category: 'timeline',
    difficulty: 'hard',
    expectedAnyCitationIds: ['EVT_0207', 'EVT_0209', 'EVT_0211', 'SYN_TIMELINE_012'],
    requiredAnyTerms: [['Điện Biên Phủ', 'Genève', 'Giơnevơ']],
  },

  // 13. Explanation — DBP significance
  {
    id: 'aq_explain_002',
    query: 'Vì sao chiến thắng Điện Biên Phủ có ý nghĩa quyết định?',
    category: 'explanation',
    difficulty: 'medium',
    expectedAnyCitationIds: ['EVT_0209', 'SYN_PERIOD_004'],
    requiredAnyTerms: [['Genève', 'Giơnevơ', 'Pháp']],
  },

  // 14. Location — Việt Minh
  {
    id: 'aq_location_001',
    query: 'Mặt trận Việt Minh được thành lập ở đâu?',
    category: 'location',
    difficulty: 'easy',
    expectedCitationIds: ['EVT_0109'],
    requiredAnyTerms: [['Pác Bó', 'Cao Bằng']],
  },

  // 15. Constitution — first VN constitution
  {
    id: 'aq_fact_002',
    query: 'Bản hiến pháp đầu tiên của nước Việt Nam Dân chủ Cộng hòa được thông qua khi nào?',
    category: 'fact',
    difficulty: 'easy',
    expectedCitationIds: ['EVT_0144'],
    requiredTerms: ['1946'],
  },

  // 16. Paris — signing date
  {
    id: 'aq_date_002',
    query: 'Hiệp định Paris được ký khi nào?',
    category: 'date',
    difficulty: 'easy',
    expectedCitationIds: ['EVT_0337'],
    requiredAnyTerms: [['27-1-1973', '27/1/1973', '27 tháng 1', '1973']],
  },

  // 17. Comparison — Paris vs US withdrawal
  {
    id: 'aq_compare_002',
    query: 'Hiệp định Paris khác gì với việc Mỹ rút quân khỏi miền Nam Việt Nam?',
    category: 'comparison',
    difficulty: 'hard',
    expectedAnyCitationIds: ['EVT_0337', 'EVT_0339'],
    requiredAnyTerms: [
      ['ký', 'hiệp định'],
      ['rút quân', 'Mỹ'],
    ],
  },

  // 18. APEC
  {
    id: 'aq_date_003',
    query: 'Việt Nam gia nhập APEC vào năm nào?',
    category: 'date',
    difficulty: 'easy',
    expectedAnyCitationIds: ['EVT_0429'],
    requiredAnyTerms: [['1998']],
  },

  // 19. Disambiguation — ASEAN vs US normalization
  {
    id: 'aq_disambig_002',
    query: 'Việt Nam gia nhập ASEAN có phải là sự kiện bình thường hóa quan hệ với Mỹ không?',
    category: 'disambiguation',
    difficulty: 'hard',
    expectedAnyCitationIds: ['EVT_0416', 'EVT_0417'],
    requiredAnyTerms: [
      ['không', 'khác', 'không phải'],
      ['ASEAN', 'Mỹ'],
    ],
  },

  // 20. False premise — Geneva permanent partition
  {
    id: 'aq_disambig_003',
    query: 'Hiệp định Genève chia cắt Việt Nam vĩnh viễn đúng không?',
    category: 'disambiguation',
    difficulty: 'hard',
    expectedAnyCitationIds: ['EVT_0212', 'EVT_0219'],
    requiredAnyTerms: [
      ['không', 'tạm thời', 'không phải'],
      ['vĩ tuyến 17', 'tổng tuyển cử'],
    ],
  },
];
