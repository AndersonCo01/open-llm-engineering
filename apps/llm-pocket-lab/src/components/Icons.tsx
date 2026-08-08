type IconProps = { size?: number; className?: string };

const Svg = ({ children, size = 24, className }: IconProps & { children: React.ReactNode }) => (
  <svg className={className} width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.9" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
    {children}
  </svg>
);

export const FlaskIcon = (props: IconProps) => <Svg {...props}><path d="M9 3h6M10 3v5l-5.5 9.5A2.3 2.3 0 0 0 6.5 21h11a2.3 2.3 0 0 0 2-3.5L14 8V3"/><path d="M7 16h10"/><circle cx="10" cy="18" r=".7" fill="currentColor" stroke="none"/><circle cx="14" cy="14" r=".7" fill="currentColor" stroke="none"/></Svg>;
export const BookIcon = (props: IconProps) => <Svg {...props}><path d="M4 5.5A3.5 3.5 0 0 1 7.5 2H11v17H7.5A3.5 3.5 0 0 0 4 22V5.5Z"/><path d="M20 5.5A3.5 3.5 0 0 0 16.5 2H13v17h3.5A3.5 3.5 0 0 1 20 22V5.5Z"/></Svg>;
export const QuizIcon = (props: IconProps) => <Svg {...props}><path d="M5 4h14v17H5z"/><path d="M9 8a3 3 0 1 1 4 2.8c-1 .4-1 1.2-1 2.2M12 17h.01"/></Svg>;
export const ChartIcon = (props: IconProps) => <Svg {...props}><path d="M5 20V10M12 20V4M19 20v-7"/></Svg>;
export const BookmarkIcon = ({ filled = false, ...props }: IconProps & { filled?: boolean }) => <Svg {...props}><path fill={filled ? "currentColor" : "none"} d="M6 3h12v18l-6-4-6 4V3Z"/></Svg>;
export const ArrowIcon = ({ direction = "right", ...props }: IconProps & { direction?: "left" | "right" }) => <Svg {...props}><path d={direction === "right" ? "M5 12h14M14 6l6 6-6 6" : "M19 12H5M10 6l-6 6 6 6"}/></Svg>;
export const CheckIcon = (props: IconProps) => <Svg {...props}><path d="m5 12 4 4L19 6"/></Svg>;
